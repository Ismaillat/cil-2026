"""Write one Kaggle-format submission CSV per individual model.

For each `preds/{name}_test.npy` shipped in the repository, decode the
softmax probabilities by median rounding and dump
`submissions/individual/{name}.csv` (columns: id, label).  These are
the files we uploaded to Kaggle to obtain the per-row Public LB scores
of Table 1.
"""
from __future__ import annotations

import glob
import os
import numpy as np
import pandas as pd


def median_round(p: np.ndarray) -> np.ndarray:
    return np.argmax(np.cumsum(p, axis=1) >= 0.5, axis=1)


def main():
    test = pd.read_csv('data/test.csv')
    paths = sorted(glob.glob('preds/*_test.npy'))
    out_dir = 'submissions/individual'
    os.makedirs(out_dir, exist_ok=True)

    print(f'{"model":<24s}  {"first 10 predictions":}')
    print('-' * 60)
    for p in paths:
        name = os.path.basename(p).replace('_test.npy', '')
        probs = np.load(p)
        pred = median_round(probs).astype(int)
        out = pd.DataFrame({'id': test['id'], 'label': pred})
        out_path = os.path.join(out_dir, f'{name}.csv')
        out.to_csv(out_path, index=False)
        print(f'{name:<24s}  {pred[:10].tolist()}   -> {out_path}')

    print(f'\nwrote {len(paths)} files to {out_dir}/')


if __name__ == '__main__':
    main()
