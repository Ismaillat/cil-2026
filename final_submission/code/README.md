# Language-Routed Specialist Ensembles for Bilingual Ordinal Sentiment

CIL 2026 project — bilingual (English + German) star-rating prediction on
product reviews, scored by `score = 1 - MAE/4`.

* **Public-leaderboard score:** `0.911`
* **Final rank:** 2 / 27
* **Authors:** Ismail Lataoui, Mehdi Hamirifou, Abdellah Janati Idrissi
  (Department of Computer Science, ETH Zurich)


## Approach in one paragraph

A pool of ten models is combined with a metric-aware meta-learner. Six are
classical / shallow baselines (TF-IDF + LogReg; frozen MiniLM + LogReg;
a Kim-style CNN; a BiLSTM; gradient boosting on TF-IDF+SVD; gradient
boosting on hand-crafted surface features); two are multilingual
transformers fine-tuned on the whole training set (XLM-R-base,
mDeBERTa-v3-base); two are **monolingual specialists** fine-tuned on
the language-filtered halves (DeBERTa-v3-large on the English reviews,
gBERT-large on the German reviews). Every model is decoded with the
**median rule**, which is Bayes-optimal under MAE on ordinal integer
labels. The aggregator we ship fits two simplex weight vectors `w_en`
and `w_de` directly minimising MAE-after-median on the validation
hold-out; at inference, the language of each review (umlaut +
function-word heuristic, the same at train and test time) selects which
weight vector is applied.


## Repository layout

```
.
├── README.md                ← this file
├── requirements.txt
├── b1.py                    ← B1  TF-IDF + LogReg
├── b2.py                    ← B2  Frozen MiniLM + LogReg
├── train_cnn_big.py         ← B3  Kim-style CNN
├── train_bilstm.py          ← B4  Bidirectional LSTM
├── b5_gbm.py                ← B5  HistGradientBoosting on TF-IDF + SVD
├── b6_surface.py            ← B6  HistGradientBoosting on surface features
├── train_model.py           ← Generic HF fine-tuner (used for the 4 transformers)
├── split_by_language.py     ← Splits train.csv into train_en.csv / train_de.csv
├── hybrid.py                ← Specialist-only language router (LB 0.910)
├── meta.py                  ← Six meta-learners + final submission CSV
├── verify_decoding.py       ← argmax vs median deltas per model (§3.1)
├── verify_per_lang.py       ← per-language ensemble scores + Jaccard (§4.3, §4.5)
├── bootstrap_ci.py          ← 95% confidence interval on the val ensemble (§5)
├── make_individual_submissions.py ← one Kaggle CSV per base model
├── make_figures.py          ← regenerates every figure used in the paper
├── eda.ipynb                ← Exploratory data analysis
├── b1_tfidf_logreg.ipynb    ← B1 notebook (same logic as b1.py)
├── b2_sentemb_logreg.ipynb  ← B2 notebook (same logic as b2.py)
├── data/
│   └── val_indices.npy      ← Canonical 10% stratified hold-out (seed 42)
└── preds/
    └── {model}_{val,test}.npy  ← softmax probabilities for each model
```

**Before running anything, download the competition CSVs from Kaggle
and place them in `data/`**:

```
data/train.csv   (id, sentence, label)
data/test.csv    (id, sentence)
```

These are not shipped in the zip (they are the competition data).

`preds/` ships pre-computed softmax outputs for all ten models on the
canonical validation split and on the test set, so the meta-learner can
be re-run in a few seconds without retraining anything. Each
`{model}_val.npy` is a `(25 200, 5)` float array and each
`{model}_test.npy` is a `(168 000, 5)` array; load them with
`numpy.load('preds/xlmr_base_val.npy')` (they are not Python source,
just `np.save`-format binary tensors).


## Quick start: reproduce the final submission

```bash
pip install -r requirements.txt
# Put the official competition CSVs in data/ first:
#   data/train.csv  (id, sentence, label)
#   data/test.csv   (id, sentence)
python meta.py --submission meta_weighted_mae_per_lang.csv
```

This reads every `preds/*_val.npy` and `preds/*_test.npy`, runs all six
aggregators (uniform average, weighted-MAE global, weighted-MAE
per-language, stacked LogReg, stacked GBM, gating MLP), prints the
5-fold OOF score of each, and writes
`submissions/meta_weighted_mae_per_lang.csv` — the file that scored
`0.911` on the public leaderboard.


## Full reproduction (from scratch)

Below is the exact sequence of commands. Each step assumes
`data/train.csv` and `data/test.csv` are present and that your
environment has the packages listed in `requirements.txt`. The four
transformer fine-tunes were run on a single 11 GB GPU (RTX 2080 Ti);
the rest run in minutes on a laptop.

### 1. Build the canonical validation split

```bash
python b1.py           # first run also writes data/val_indices.npy
```

`b1.py` creates `data/val_indices.npy` (10 % stratified, seed 42) on
its first invocation; every other script reads the same file, so all
models are scored on exactly the same 25 200 reviews.

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

### 6. Specialist-only hybrid (optional baseline)

```bash
python hybrid.py
```

For every review, routes to the specialist that matches its detected
language (English → DeBERTa-v3-large EN, German → gBERT-large DE),
decodes by median, and writes
`submissions/hybrid_deberta_en_gbert_large_de.csv`. Reproduces the
"Hybrid (specialists only)" row of Table 1 (val 0.911 / LB 0.910).

### 7. Meta-learner

```bash
python meta.py --submission meta_weighted_mae_per_lang.csv
```

`meta.py` auto-discovers every `preds/*_val.npy` it finds, evaluates
all six aggregators by 5-fold stratified OOF on the validation set,
refits the chosen one on the full hold-out, and writes the final
submission CSV under `submissions/`. The aggregator that produced the
leaderboard submission is **`weighted-MAE per-language`**.


## Reproducing each table and figure of the paper

Every numerical claim in the report can be regenerated from
`preds/*.npy` without retraining:

| Result in the paper | Command |
|---|---|
| **Table 1**, *Val* + *EN/DE* columns for the 10 base models | `python make_figures.py` (prints per-language scores) |
| **Table 1**, row *Hybrid (specialists only)* | `python hybrid.py` → `submissions/hybrid_deberta_en_gbert_large_de.csv` (val 0.911 / EN 0.914 / DE 0.908) |
| **Table 1**, row *Weighted-MAE per-lang* (Val / EN / DE / LB) | `python verify_per_lang.py` (prints 0.913 / 0.915 / 0.911) and `python meta.py` (LB 0.911 CSV) |
| **Table 1**, *Public LB* column for each base model | `python make_individual_submissions.py` (writes 10 CSVs under `submissions/individual/`, one per model) |
| **Table 2**, all six OOF rows | `python meta.py` (Uniform 0.9066, weighted_mae 0.9121, weighted_mae_per_lang 0.9121, LogReg 0.9116, GBM 0.9112, gating_mlp 0.9072) |
| **Section 3.1**, argmax → median deltas | `python verify_decoding.py` |
| **Section 4.1**, XLM-R+mDeBERTa uniform 0.908 | `python verify_per_lang.py` |
| **Section 4.3**, per-language weights $w_{\text{en}}, w_{\text{de}}$ | `python meta.py` prints the two simplex vectors |
| **Section 4.3**, Jaccard 0.60 on val_DE between DeBERTa-EN and mDeBERTa | `python verify_per_lang.py` |
| **Section 4.4**, Jaccard matrix on whole val | `python make_figures.py` (prints the 10x10 matrix) |
| **Section 4.5**, 0↔4 confusion rate per language | `python verify_per_lang.py` (EN 0.09 %, DE 0.12 %) |
| **Section 5**, 95% bootstrap CI [0.9115, 0.9148] | `python bootstrap_ci.py` |
| **Figures 1–5** | `python make_figures.py` (writes to `figs/`) |

The full sequence to regenerate everything:

```bash
python meta.py                          # Tables 1 (final row), 2; Figure 2 weights; meta_*.csv
python hybrid.py                        # Table 1 hybrid row
python verify_decoding.py               # Section 3.1 deltas
python verify_per_lang.py               # Table 1 per-lang row, §4.3 Jaccard, §4.5 confusion
python bootstrap_ci.py                  # Section 5 confidence interval
python make_individual_submissions.py   # 10 per-model Kaggle CSVs
python make_figures.py                  # all six PDF figures + §4.4 Jaccard matrix
```

Everything runs in under five minutes on CPU (the gating MLP inside
`meta.py` benefits from a GPU; it is skipped gracefully if the cached
MiniLM embeddings are absent).


## Notes on reproducibility

* All models use `seed=42`. The validation split (`data/val_indices.npy`)
  is fixed across runs, so the `_val.npy` softmax files in `preds/` are
  directly comparable.
* The language heuristic (`split_by_language.py::is_german`,
  `meta.py::detect_language`, `hybrid.py::is_german`) is identical at
  training time and at inference time; reviews that look German at
  training time also look German at test time, so the two specialists
  are queried on the same population they were trained on.
* For the two multilingual transformers, mixed-precision training and a
  single epoch are sufficient to land on the reported validation score
  to within ±0.001; the third epoch tightens the calibration that the
  meta-learner exploits.
* `make_figures.py` regenerates every figure from `preds/*.npy` and
  `data/train.csv`. The numbers it prints (per-language single-model
  scores, Jaccard error overlaps) match Table 1 / Section 4.4 of the
  report to the millième.
* The gating MLP (`[5]` in `meta.py`) needs both the cached MiniLM
  embeddings (re-created by `b2.py`) and ideally a CUDA GPU; the other
  five aggregators run in under a minute on CPU and are the path used
  to produce the submitted CSV.


## Verifying

Without retraining anything, running `python meta.py` should print:

```
UNIFORM AVG / median               score=0.9066
weighted_mae (OOF)                 score=0.9121  (per-fold std 0.0015)
weighted_mae_per_lang (OOF)        score=0.9121  (per-fold std 0.0017)
logreg (OOF)                       score=0.9116  (per-fold std 0.0012)
gbm (OOF)                          score=0.9112  (per-fold std 0.0013)
gating_mlp (OOF)                   score=0.9072  (per-fold std 0.0011)
```

— which is exactly Table 2 of the paper. `python hybrid.py` reproduces
the "Hybrid (specialists only)" row of Table 1 (val 0.911, EN 0.914,
DE 0.908).


## Outputs

After `meta.py` finishes, `submissions/` contains:

```
meta_weighted_mae_per_lang.csv   ← language-routed simplex weights  (LB 0.911, the one we submitted)
meta_weighted_mae.csv            ← global simplex weights
meta_logreg.csv                  ← stacked LogReg
meta_gbm.csv                     ← stacked HistGradientBoosting
meta_gating.csv                  ← gating MLP (Wasserstein-1 loss, 3-seed avg)
```

The validation scores printed by `meta.py` should reproduce the
numbers in Table 2 of the paper to within the per-fold standard
deviation.


## License

The code in this repository is released under the MIT license. The
pre-trained models we fine-tune are distributed under their own
licenses on the Hugging Face Hub.
