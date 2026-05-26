# CIL 2026 — Bilingual Sentiment Classification

Predict the star rating (`0`–`4`) of bilingual (German/English) product reviews.
Metric: `score = 1 − MAE/4`.

## Final approach + result

An **ensemble of two fine-tuned multilingual encoders**, averaged in
probability space and decoded with the **MAE-optimal median rule**:

| Method | Val score |
|---|---|
| B1: TF-IDF + Logistic Regression | 0.882 |
| B2: Frozen multilingual sentence encoder + LogReg (linear probe) | ~0.85 |
| XLM-RoBERTa-base (fine-tuned) | 0.905 |
| mDeBERTa-v3-base (fine-tuned) | 0.907 |
| **Ensemble (ours)** | **0.908** |

Hard baseline for grade 6 on the Kaggle public LB: **0.906**. The ensemble's
public-LB score matches the val score to 3 decimals, indicating no
val-set overfitting.

## Project layout

```
code/
├── README.md                  this file
├── requirements.txt           pinned versions (transformers==4.57.6)
├── .gitignore
├── data/                      train.csv, test.csv, val_indices.npy (not committed)
├── preds/                     softmax probabilities per model (committed)
│   ├── xlmr_base_val.npy      (25200, 5) — val predictions
│   ├── xlmr_base_test.npy     (168000, 5) — test predictions
│   ├── mdeberta_val.npy
│   └── mdeberta_test.npy
├── submissions/
│   └── ensemble.csv           Kaggle submission produced by ensemble.py
├── eda.ipynb                  exploratory data analysis (run locally on a Mac)
├── b1_tfidf_logreg.ipynb      B1 baseline; also creates data/val_indices.npy
├── b2_sentemb_logreg.ipynb    B2 baseline (frozen sentence encoder linear probe)
├── train_model.py             unified trainer; parameterized by --model and --name
├── ensemble.py                averages preds/, applies median decoding, writes submission
├── run.slurm                  generic SLURM wrapper (sbatch run.slurm <python args>)
└── report/                    LaTeX project (ICML template) — report.tex, report.bib, *.sty
```

## Setup — macOS

```bash
git clone git@github.com:Ismaillat/cil-2026.git cil
cd cil
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Bring the data:

```bash
mkdir -p data
scp <nethz>@student-cluster.inf.ethz.ch:/cluster/courses/cil/text-classification/data/train.csv data/
scp <nethz>@student-cluster.inf.ethz.ch:/cluster/courses/cil/text-classification/data/test.csv  data/
```

## Setup — cluster (ETH student cluster)

One-time `~/.bashrc` block (course-provided):

```bash
__conda_setup="$('/cluster/courses/cil/envs/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
[ $? -eq 0 ] && eval "$__conda_setup"
module load cuda/12.6.0
conda activate /cluster/courses/cil/envs/envs/text-classification
```

User-space packages the course env is missing (do this once):

```bash
/cluster/courses/cil/envs/envs/text-classification/bin/pip install --user \
    'transformers==4.57.6' protobuf sentencepiece tiktoken
```

The transformers pin is required — the cluster's default (5.x preview) crashes
on the mDeBERTa-v3 SentencePiece file. 4.57.6 loads it correctly.

Symlink the data so notebooks find it:

```bash
cd ~/cil
mkdir -p data submissions preds runs
ln -sf /cluster/courses/cil/text-classification/data/train.csv data/train.csv
ln -sf /cluster/courses/cil/text-classification/data/test.csv  data/test.csv
```

## Reproducing the ensemble result

### From cached predictions (no GPU needed, ~10 seconds)

`preds/*.npy` are committed. The ensemble runs entirely on CPU:

```bash
python ensemble.py
# prints per-model val scores, the ENSEMBLE val score (0.908), the confusion
# matrix, and writes submissions/ensemble.csv
```

### From scratch on the cluster (GPU, ~3 + ~3 hours sequentially)

The cluster QOS limits each user to one running GPU job at a time, so the
two trainings run back-to-back.

```bash
cd ~/cil

# 1. Create the canonical 10% val split (one-liner from b1_tfidf_logreg.ipynb)
python -c "
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
train = pd.read_csv('data/train.csv')
_, val_idx = train_test_split(np.arange(len(train)), test_size=0.1,
                              stratify=train['label'], random_state=42)
np.save('data/val_indices.npy', val_idx)
"

# 2. Train each model (saves only the .npy probabilities, no weights checkpoint)
sbatch run.slurm train_model.py --model xlm-roberta-base       --name xlmr_base
sbatch run.slurm train_model.py --model microsoft/mdeberta-v3-base --name mdeberta --batch 16

# 3. Watch
squeue -u $USER
tail -f runs/slurm-<JOBID>.err   # progress bar
grep "'loss'" runs/slurm-<JOBID>.out | tail   # loss values

# 4. After both finish: build the ensemble submission
/cluster/courses/cil/envs/envs/text-classification/bin/python ensemble.py
```

The cluster uses a `nodelist` of 2080 Ti nodes in `run.slurm` (the 5060 Ti
nodes are incompatible with the course env's PyTorch).

## Extending the ensemble

`ensemble.py` auto-globs every `preds/*_val.npy` / `preds/*_test.npy` pair, so
adding a third model is as simple as training it and re-running:

```bash
sbatch run.slurm train_model.py --model xlm-roberta-base --name xlmr_base_s1 --seed 1
# ... wait for it ...
python ensemble.py
```

### Building a meta-model on top

The committed `preds/*.npy` give you everything needed to train a stacked
meta-learner without re-training the base models:

```python
import numpy as np, pandas as pd
val_idx = np.load('data/val_indices.npy')
y_val = pd.read_csv('data/train.csv').loc[val_idx, 'label'].values

# stack model probabilities side by side: (n, 5 * n_models)
val_X  = np.concatenate([np.load('preds/xlmr_base_val.npy'),
                         np.load('preds/mdeberta_val.npy')], axis=1)
test_X = np.concatenate([np.load('preds/xlmr_base_test.npy'),
                         np.load('preds/mdeberta_test.npy')], axis=1)

# train any meta-learner on (val_X, y_val), predict on test_X
# e.g. LogisticRegression, GradientBoosting, a small MLP, etc.
```

Decode the meta-model's test probabilities with the same `median_round` rule
used in `ensemble.py` for an MAE-optimal submission.

## Model weights

We do **not** ship the fine-tuned weights. They were deleted during cluster
disk-quota cleanup, and `save_strategy='no'` is set on subsequent runs to
avoid recurring quota issues. The pipeline writes only the small `.npy`
probability arrays, which are all the downstream stages (ensembling,
plotting, report tables) actually need. To regenerate weights, retrain from
scratch with `train_model.py`.

## Report

The 4-page IEEE/ICML-style report lives in `report/`. Compile with `latexmk`:

```bash
cd report
latexmk -pdf report.tex
# -> report/report.pdf
```

In Overleaf: upload the contents of `report/` and set the main document to
`report.tex`.

## Authors

- Ismail Lataoui (ilataoui@ethz.ch)
- Mehdi Hamirifou
- Abdellah Janati Idrissi

## Declaration of originality

This work is our own. We used the publicly available pre-trained models
`xlm-roberta-base`, `microsoft/mdeberta-v3-base`, and
`paraphrase-multilingual-MiniLM-L12-v2`, and the libraries `scikit-learn`,
`PyTorch`, `transformers`, and `sentence-transformers`. An AI assistant
(Anthropic Claude) was used during development for coding help and
documentation review, per ETH policy on AI-assisted work.
