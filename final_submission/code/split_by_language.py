"""
Split the training set into English-only and German-only subsets.

The 10% canonical hold-out (val_indices) is EXCLUDED from both files so
that any specialist trained on these files can still be evaluated on the
canonical val without leakage.  The language heuristic is the same as in
the EDA notebook: a review is tagged German if it contains any umlaut
(ä, ö, ü, ß) or any common German function word.

Produces:
    data/train_en.csv  (~115k rows)   English-only training pool
    data/train_de.csv  (~110k rows)   German-only training pool
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


_DE_CHARS = ('ä', 'ö', 'ü', 'ß')
_DE_WORDS = (' der ', ' die ', ' das ', ' und ', ' ich ', ' nicht ',
             ' ist ', ' ein ', ' eine ', ' mit ', ' für ',
             ' auch ', ' war ', ' es ', ' zu ', ' den ')


def is_german(text: str) -> bool:
    t = (text or '').lower()
    if any(c in t for c in _DE_CHARS):
        return True
    padded = f' {t} '
    return any(w in padded for w in _DE_WORDS)


def main():
    train = pd.read_csv('data/train.csv')
    val_idx = np.load('data/val_indices.npy')
    train_only = train.drop(index=val_idx).reset_index(drop=True)

    mask_de = train_only['sentence'].apply(is_german)
    train_de = train_only[mask_de].reset_index(drop=True)
    train_en = train_only[~mask_de].reset_index(drop=True)

    os.makedirs('data', exist_ok=True)
    train_en.to_csv('data/train_en.csv', index=False)
    train_de.to_csv('data/train_de.csv', index=False)

    n = len(train_only)
    print(f'total (train - val):  {n}')
    print(f'  train_en: {len(train_en):6d} rows  '
          f'({100 * len(train_en) / n:.1f}%)')
    print(f'  train_de: {len(train_de):6d} rows  '
          f'({100 * len(train_de) / n:.1f}%)')
    print('per-label distribution (en | de):')
    en_dist = train_en['label'].value_counts().sort_index()
    de_dist = train_de['label'].value_counts().sort_index()
    for lbl in range(5):
        print(f'  label={lbl}:  en={en_dist.get(lbl, 0):6d}  '
              f'de={de_dist.get(lbl, 0):6d}')


if __name__ == '__main__':
    main()
