"""
B1 — TF-IDF + Logistic Regression.

Reproduces `b1_tfidf_logreg.ipynb` as a script and saves the val and
test softmax probabilities to `preds/b1_{val,test}.npy` so that
`meta.py` can pick them up.

On a first run it also writes `data/val_indices.npy` (the canonical
10% stratified hold-out, seed 42) which every other model in the
pipeline then reads — that way every model is evaluated on exactly
the same review IDs and validation scores are directly comparable.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


SEED = 42
VAL_SIZE = 0.10


def median_round(p: np.ndarray) -> np.ndarray:
    return np.argmax(np.cumsum(p, axis=1) >= 0.5, axis=1)


def main():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    val_path = 'data/val_indices.npy'
    if os.path.exists(val_path):
        val_idx = np.load(val_path)
    else:
        _, val_idx = train_test_split(
            np.arange(len(train)), test_size=VAL_SIZE,
            stratify=train['label'], random_state=SEED,
        )
        np.save(val_path, val_idx)

    train_df = train.drop(index=val_idx).reset_index(drop=True)
    val_df = train.loc[val_idx].reset_index(drop=True)

    vec = TfidfVectorizer(
        ngram_range=(1, 2), max_features=50_000,
        min_df=2, sublinear_tf=True,
    )
    X_train = vec.fit_transform(train_df['sentence'])
    X_val = vec.transform(val_df['sentence'])
    X_test = vec.transform(test['sentence'])
    y_train = train_df['label'].to_numpy()
    y_val = val_df['label'].to_numpy()

    clf = LogisticRegression(C=1.0, max_iter=200, random_state=SEED)
    clf.fit(X_train, y_train)

    val_probs = clf.predict_proba(X_val)
    test_probs = clf.predict_proba(X_test)

    os.makedirs('preds', exist_ok=True)
    np.save('preds/b1_val.npy', val_probs)
    np.save('preds/b1_test.npy', test_probs)

    for tag, pred in [('argmax', np.argmax(val_probs, 1)),
                      ('median', median_round(val_probs))]:
        mae = mean_absolute_error(y_val, pred)
        print(f'[b1 {tag}] score={1 - mae / 4:.4f}  mae={mae:.4f}  '
              f'acc={(y_val == pred).mean():.4f}')


if __name__ == '__main__':
    main()
