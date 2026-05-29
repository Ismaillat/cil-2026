"""Reproduce the argmax-vs-median deltas reported in Section 3.1.

For every cached prediction file in preds/, decode the validation
probabilities with both rules and print the score gain that switching
to the median brings. The numbers backing the sentence

    "switching from argmax to the median lifts the validation score of
    every fine-tuned encoder by between 0.0004 and 0.0013; on the
    linear baselines, by +0.005 to +0.007."

(Section 3.1 of the report) should be reproduced exactly.
"""
from __future__ import annotations

import glob
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def median_round(p: np.ndarray) -> np.ndarray:
    return np.argmax(np.cumsum(p, axis=1) >= 0.5, axis=1)


def score(y, yhat):
    return 1 - mean_absolute_error(y, yhat) / 4


def main():
    train = pd.read_csv('data/train.csv')
    val_idx = np.load('data/val_indices.npy')
    y_val = train.loc[val_idx, 'label'].to_numpy()

    paths = sorted(glob.glob('preds/*_val.npy'))
    if not paths:
        raise SystemExit('no preds/*_val.npy found')

    print(f'{"model":<24s}  {"argmax":>8s}  {"median":>8s}  {"Δ":>8s}')
    print('-' * 56)
    rows = []
    for p in paths:
        name = os.path.basename(p).replace('_val.npy', '')
        probs = np.load(p)
        s_arg = score(y_val, np.argmax(probs, axis=1))
        s_med = score(y_val, median_round(probs))
        delta = s_med - s_arg
        print(f'{name:<24s}  {s_arg:8.4f}  {s_med:8.4f}  {delta:+8.4f}')
        rows.append((name, s_arg, s_med, delta))

    # Aggregate by family to verify the two ranges quoted in the paper.
    fine_tuned = {'xlmr_base', 'mdeberta',
                  'deberta_v3_large_en', 'gbert_large_de'}
    linear = {'b1', 'b2'}

    def report(names, label):
        deltas = [d for n, _, _, d in rows if n in names]
        if deltas:
            print(f'{label}: min={min(deltas):+.4f}  '
                  f'max={max(deltas):+.4f}  mean={np.mean(deltas):+.4f}')

    print()
    report(fine_tuned, 'fine-tuned encoders (report: +0.0003 to +0.0013)')
    report(linear,     'linear baselines    (report: +0.0055 to +0.0067)')


if __name__ == '__main__':
    main()
