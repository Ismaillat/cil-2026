"""
B5 — Histogram Gradient Boosting on TF-IDF features.

A tree-based ensemble baseline that complements the linear (B1, B2),
convolutional (B3) and recurrent (B4) families with a fundamentally
different inductive bias: greedy axis-aligned splits on sparse lexical
features.  This maximises error-set diversity for the meta-learner.

The model trains on a dense projection of the TF-IDF features:
HistGradientBoosting in sklearn does not accept sparse inputs, so we
keep the same vectoriser as B1 (50K ngrams, sublinear TF) and project
it down with TruncatedSVD to 200 dense components before fitting.

Output: preds/b5_gbm_{val,test}.npy
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


SEED = 42
VAL_SIZE = 0.10
N_FEATURES_TFIDF = 50_000
N_SVD_COMPONENTS = 200


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
    y_train = train_df['label'].to_numpy()
    y_val = val_df['label'].to_numpy()

    print(f'fitting TF-IDF ...', flush=True)
    vec = TfidfVectorizer(
        ngram_range=(1, 2), max_features=N_FEATURES_TFIDF,
        min_df=2, sublinear_tf=True,
    )
    X_train_sp = vec.fit_transform(train_df['sentence'])
    X_val_sp = vec.transform(val_df['sentence'])
    X_test_sp = vec.transform(test['sentence'])
    print(f'  TF-IDF shape: train={X_train_sp.shape}', flush=True)

    print(f'projecting to {N_SVD_COMPONENTS}-d with TruncatedSVD ...', flush=True)
    svd = TruncatedSVD(n_components=N_SVD_COMPONENTS, random_state=SEED)
    X_train = svd.fit_transform(X_train_sp).astype(np.float32)
    X_val = svd.transform(X_val_sp).astype(np.float32)
    X_test = svd.transform(X_test_sp).astype(np.float32)
    print(f'  dense shape: train={X_train.shape}  '
          f'explained variance: {svd.explained_variance_ratio_.sum():.3f}',
          flush=True)

    print(f'fitting HistGradientBoosting ...', flush=True)
    clf = HistGradientBoostingClassifier(
        max_iter=500, max_depth=6, learning_rate=0.05,
        l2_regularization=1.0, random_state=SEED, early_stopping=False,
    )
    clf.fit(X_train, y_train)
    print('  done', flush=True)

    val_probs = clf.predict_proba(X_val)
    test_probs = clf.predict_proba(X_test)

    os.makedirs('preds', exist_ok=True)
    np.save('preds/b5_gbm_val.npy', val_probs)
    np.save('preds/b5_gbm_test.npy', test_probs)

    for tag, pred in [('argmax', np.argmax(val_probs, 1)),
                      ('median', median_round(val_probs))]:
        mae = mean_absolute_error(y_val, pred)
        print(f'[b5_gbm {tag}] score={1 - mae / 4:.4f}  mae={mae:.4f}  '
              f'acc={(y_val == pred).mean():.4f}', flush=True)


if __name__ == '__main__':
    main()
