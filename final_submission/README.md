# Language-Routed Specialist Ensembles for Bilingual Ordinal Sentiment

CIL 2026 project — bilingual (English + German) star-rating prediction on
product reviews, scored by `score = 1 - MAE/4`.

* **Public-leaderboard score:** `0.91108`
* **Final rank:** 2 / 27
* **Authors:** Ismail Lataoui, Mehdi Hamirifou, Abdellah Janati Idrissi
  (Department of Computer Science, ETH Zurich)


## Approach in one paragraph

A pool of ten models is combined with a metric-aware meta-learner. Six are
classical / shallow baselines (TF-IDF + LogReg; frozen MiniLM + LogReg;
a Kim-style CNN; a BiLSTM; gradient boosting on TF-IDF+SVD; gradient
boosting on hand-crafted surface features); two are multilingual
transformers fine-tuned on the whole training set (XLM-R-base,
mDeBERTa-v3-base); two are **monolingual specialists**
fine-tuned on the language-filtered halves (DeBERTa-v3-large on the
English reviews, gBERT-large on the German reviews). Every model is
decoded with the **median rule**, which is Bayes-optimal under MAE on
ordinal integer labels. The aggregator we ship fits two simplex weight
vectors `w_en` and `w_de` directly minimising MAE-after-median on the
validation hold-out; at inference, the language of each review (umlaut +
function-word heuristic, the same at train and test time) selects which
weight vector is applied. See `report/report.tex` for details.


## Repository layout

```
final_submission/
├── README.md                ← this file
├── report/
│   ├── report.tex           ← 4-page workshop-style paper
│   ├── report.bib           ← BibTeX references
│   ├── make_figures.py      ← regenerates every figure used in the paper
│   ├── figs/                ← .pdf figures used by report.tex
│   └── *.sty / *.bst        ← ICML 2024 LaTeX style files
└── code/
    ├── requirements.txt
    ├── run.slurm            ← submission script for the ETH student cluster
    ├── b1.py                ← B1  TF-IDF + LogReg
    ├── b2.py                ← B2  Frozen MiniLM + LogReg
    ├── train_cnn_big.py     ← B3  Kim-style CNN
    ├── train_bilstm.py      ← B4  Bidirectional LSTM
    ├── b5_gbm.py            ← B5  HistGradientBoosting on TF-IDF + SVD
    ├── b6_surface.py        ← B6  HistGradientBoosting on surface features
    ├── train_model.py       ← Generic HF fine-tuner (used for the 4 transformers)
    ├── split_by_language.py ← Splits train.csv into train_en.csv / train_de.csv
    ├── meta.py              ← Five meta-learners + final submission CSV
    ├── eda.ipynb            ← Exploratory data analysis
    ├── b1_tfidf_logreg.ipynb
    ├── b2_sentemb_logreg.ipynb
    ├── data/
    │   └── val_indices.npy  ← Canonical 10% stratified hold-out (seed 42)
    └── preds/
        └── {model}_{val,test}.npy  ← softmax probabilities for each model
```

`preds/` ships pre-computed softmax outputs for all ten models on the
canonical validation split and on the test set, so the meta-learner can
be re-run in a few seconds without retraining anything.


## Quick start: reproduce the final submission

```bash
cd code/
pip install -r requirements.txt              # see requirements.txt
# Put the official competition CSVs in data/ first:
#   data/train.csv  (sentence,label,id)
#   data/test.csv   (sentence,id)
python meta.py --submission meta_weighted_mae_per_lang.csv
```

This reads every `preds/*_val.npy` and `preds/*_test.npy`, runs all five
aggregators (uniform average, weighted-MAE global, weighted-MAE
per-language, stacked LogReg, stacked GBM), prints the 5-fold OOF score
of each, and writes `submissions/meta_weighted_mae_per_lang.csv` — the
file that scored `0.91108` on the public leaderboard.


## Full reproduction (from scratch)

Below is the exact sequence of commands. Each step assumes you are in
`code/`, that `data/train.csv` and `data/test.csv` are present, and that
your environment has the packages listed in `requirements.txt`. The four
transformer fine-tunes were run on a single 11 GB GPU (RTX 2080 Ti) on
the ETH student cluster; the rest run in minutes on a laptop.

### 1. Build the canonical validation split

```bash
python b1.py           # first run also writes data/val_indices.npy
```

`b1.py` creates `data/val_indices.npy` (10 % stratified, seed 42) on its
first invocation; every other script reads the same file, so all models
are scored on exactly the same 25 200 reviews.

### 2. Classical / shallow baselines (CPU, ~minutes each)

```bash
python b1.py            # TF-IDF + LogReg
python b2.py            # frozen MiniLM + LogReg
python b5_gbm.py        # HistGradientBoosting on TF-IDF + SVD-200
python b6_surface.py    # HistGradientBoosting on 19 surface features
```

### 3. Neural baselines (single-GPU, ~10–20 min each)

```bash
python train_cnn_big.py  --name cnn_big      # B3
python train_bilstm.py   --name bilstm       # B4
```

### 4. Multilingual transformers (single 11 GB GPU)

```bash
python train_model.py \
    --model xlm-roberta-base --name xlmr_base \
    --epochs 3 --batch 16 --lr 2e-5 --max_length 256

python train_model.py \
    --model microsoft/mdeberta-v3-base --name mdeberta \
    --epochs 3 --batch 16 --lr 2e-5 --max_length 256 --slow_tokenizer
```

`--slow_tokenizer` is needed for mDeBERTa-v3 on `transformers==4.57.6`.

### 5. Monolingual specialists

First split the training pool by language (the canonical val rows are
excluded from both halves):

```bash
python split_by_language.py
# -> data/train_en.csv  (~117k rows)
# -> data/train_de.csv  (~110k rows)
```

Then fine-tune one specialist on each half:

```bash
python train_model.py \
    --model microsoft/deberta-v3-large --name deberta_v3_large_en \
    --train_file data/train_en.csv \
    --epochs 2 --batch 8 --grad_accum 8 \
    --lr 1e-5 --warmup_ratio 0.1 --max_length 192 \
    --gradient_checkpointing

python train_model.py \
    --model deepset/gbert-large --name gbert_large_de \
    --train_file data/train_de.csv \
    --epochs 2 --batch 8 --grad_accum 8 \
    --lr 1e-5 --warmup_ratio 0.1 --max_length 192 \
    --gradient_checkpointing
```

`--gradient_checkpointing` trades a bit of compute for memory; on the
2080 Ti these two fine-tunes take ~14 h each.

### 6. Meta-learner

```bash
python meta.py --submission meta_weighted_mae_per_lang.csv
```

`meta.py` auto-discovers every `preds/*_val.npy` it finds, evaluates all
five aggregators by 5-fold stratified OOF on the validation set, refits
the chosen one on the full hold-out, and writes the final submission CSV
under `submissions/`. The aggregator that produced the leaderboard
submission is **`weighted-MAE per-language`**.


## On the ETH student cluster

The included `run.slurm` is a thin wrapper that lets you submit any of
the above commands as a single-GPU job:

```bash
sbatch run.slurm train_model.py --model xlm-roberta-base --name xlmr_base \
       --epochs 3 --batch 16 --lr 2e-5 --max_length 256
```

It loads CUDA 12.6, points at the course-provided text-classification
Python environment, and requests one of the 2080 Ti nodes. Adjust the
`--time` and `--nodelist` lines for your own setup.


## Notes on reproducibility

* All models use `seed=42`. The validation split (`data/val_indices.npy`)
  is fixed across runs, so the `_val.npy` softmax files in `preds/` are
  directly comparable.
* The language heuristic (`split_by_language.py::is_german`,
  `meta.py::detect_language`, `b6_surface.py`) is identical at training
  time and at inference time; reviews that look German at train time
  also look German at test time, so the two specialists are queried on
  the same population they were trained on.
* For the two multilingual transformers, mixed-precision training and a
  single epoch are sufficient to land on the reported validation score
  to within ±0.001; the third epoch tightens the calibration that the
  meta-learner exploits.


## Outputs

After `meta.py` finishes, `submissions/` contains:

```
meta_weighted_mae_per_lang.csv   ← language-routed simplex weights  (LB 0.91108, the one we submitted)
meta_weighted_mae.csv            ← global simplex weights
meta_logreg.csv                  ← stacked LogReg
meta_gbm.csv                     ← stacked HistGradientBoosting
meta_gating.csv                  ← gating MLP (Wasserstein-1 loss, 3-seed avg)
```

The validation scores printed by `meta.py` should reproduce the numbers
in Table 2 of the paper to within the per-fold standard deviation.


## License

The code in this repository is released under the MIT license. The
pre-trained models we fine-tune are distributed under their own licenses
on the Hugging Face Hub.
