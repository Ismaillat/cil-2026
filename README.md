# CIL 2026 — Sentiment Classification

Bilingual (DE/EN) product-review star-rating prediction.

## Setup (macOS)

### 1. Prerequisites

- macOS with **Python 3.9+** (`python3 --version` to check)
- Git
- An nethz account with cluster access (to grab the data)

### 2. Clone

```bash
git clone <repo-url> cil
cd cil
```

### 3. Create the virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Activate the env every new terminal session: `source .venv/bin/activate`.

### 4. Get the data

The CSVs are not committed (too large). Copy them from the cluster:

```bash
mkdir -p data
scp <your-nethz>@student-cluster.inf.ethz.ch:/cluster/courses/cil/text-classification/data/train.csv data/
scp <your-nethz>@student-cluster.inf.ethz.ch:/cluster/courses/cil/text-classification/data/test.csv  data/
```

You should end up with `data/train.csv` (~58 MB) and `data/test.csv` (~38 MB).

### 5. Verify

```bash
python -c "
import pandas as pd
print(pd.read_csv('data/train.csv').shape)
"
```

Expect: `(252000, 3)`.

## Project layout

```
code/
├── data/                          # CSVs + cached features (gitignored)
├── submissions/                   # Kaggle CSVs
├── eda.ipynb                      # exploratory data analysis
├── b1_tfidf_logreg.ipynb          # baseline 1: TF-IDF + LogReg
├── b2_sentemb_logreg.ipynb        # baseline 2: frozen sentence-transformer + LogReg
├── requirements.txt
└── README.md
```

## Running notebooks

In VS Code: open the `.ipynb`, click **Select Kernel** → pick the one at `.venv/bin/python`.

In Jupyter: `jupyter notebook <file>.ipynb`.

Always run `b1_tfidf_logreg.ipynb` first — it creates `data/val_indices.npy`, the canonical 90/10 split that every later notebook reuses.

## Notes

- The first run of `b2_sentemb_logreg.ipynb` downloads ~470 MB of model weights and encodes all 420K reviews. Encoding is cached to `data/b2_*.npy` so subsequent runs are instant.
- On Apple Silicon (M1+) the encoder uses MPS automatically. On Intel Macs it falls back to CPU (~2 h for the encode).
- The cluster is for the main transformer fine-tune, not the baselines. Separate doc.
