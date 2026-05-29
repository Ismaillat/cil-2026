"""Specialist-only hybrid: route each review to its native specialist.

For every review we pick the prediction of the monolingual specialist
whose language matches that of the review (English -> DeBERTa-v3-large EN,
German -> gBERT-large DE), then decode with the MAE-optimal median rule.

No learned weights, no other models -- just a clean baseline showing
that language-conditioned routing alone already moves the validation
score above the strongest single multilingual transformer (mDeBERTa,
0.907). This is the row labelled "Hybrid (specialists only)" in
Table 1 of the report (LB 0.910).

Outputs:
    submissions/hybrid_deberta_en_gbert_large_de.csv
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


_DE_CHARS = ('ä', 'ö', 'ü', 'ß')
_DE_WORDS = (' der ', ' die ', ' das ', ' und ', ' ich ', ' nicht ',
             ' ist ', ' ein ', ' eine ', ' mit ', ' für ', ' auch ',
             ' war ', ' es ', ' zu ', ' den ')


def is_german(text: str) -> bool:
    t = (text or '').lower()
    if any(c in t for c in _DE_CHARS):
        return True
    return any(w in f' {t} ' for w in _DE_WORDS)


def detect_language(texts):
    return np.array(['de' if is_german(t) else 'en' for t in texts])


def median_round(p):
    return np.argmax(np.cumsum(p, axis=1) >= 0.5, axis=1)


def route(en_probs, de_probs, lang):
    """Pick row i from en_probs if lang[i]=='en', else from de_probs."""
    out = np.empty_like(en_probs)
    en_mask = (lang == 'en')
    out[en_mask] = en_probs[en_mask]
    out[~en_mask] = de_probs[~en_mask]
    return out


def main():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    val_idx = np.load('data/val_indices.npy')

    val_df = train.loc[val_idx].reset_index(drop=True)
    y_val = val_df['label'].to_numpy()
    lang_val = detect_language(val_df['sentence'])
    lang_test = detect_language(test['sentence'])

    en_val = np.load('preds/deberta_v3_large_en_val.npy')
    de_val = np.load('preds/gbert_large_de_val.npy')
    en_test = np.load('preds/deberta_v3_large_en_test.npy')
    de_test = np.load('preds/gbert_large_de_test.npy')

    val_probs = route(en_val, de_val, lang_val)
    test_probs = route(en_test, de_test, lang_test)

    pred_val = median_round(val_probs)
    mae = mean_absolute_error(y_val, pred_val)
    print(f'val score = {1 - mae / 4:.4f}   mae = {mae:.4f}   '
          f'acc = {(y_val == pred_val).mean():.4f}')

    mask_en = (lang_val == 'en')
    mae_en = mean_absolute_error(y_val[mask_en], pred_val[mask_en])
    mae_de = mean_absolute_error(y_val[~mask_en], pred_val[~mask_en])
    print(f'  val EN ({mask_en.sum()}): score = {1 - mae_en / 4:.4f}')
    print(f'  val DE ({(~mask_en).sum()}): score = {1 - mae_de / 4:.4f}')

    os.makedirs('submissions', exist_ok=True)
    out = pd.DataFrame({'id': test['id'],
                        'label': median_round(test_probs).astype(int)})
    out_path = 'submissions/hybrid_deberta_en_gbert_large_de.csv'
    out.to_csv(out_path, index=False)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
