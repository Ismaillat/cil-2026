"""Per-language sanity checks for Table 1 and Section 4.3 of the report.

Prints:

  * the full-fit per-language weighted-MAE ensemble scores on val_EN
    and val_DE -- these back the row "Weighted-MAE per-lang" of
    Table 1 (val 0.912, EN 0.915, DE 0.909).
  * the off-diagonal 0<->4 confusion rates of the per-language
    ensemble (Section 4.5: "stays below 0.5% on each side").
  * the Jaccard error overlap between every pair of models on the
    val_EN partition and on the val_DE partition separately. In
    particular the cell DeBERTa-EN x mDeBERTa on val_DE backs the
    "Jaccard 0.55" claim in Section 4.3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_absolute_error

import meta


def median_round(p):
    return np.argmax(np.cumsum(p, axis=1) >= 0.5, axis=1)


def main():
    train = pd.read_csv('data/train.csv')
    val_idx = np.load('data/val_indices.npy')
    y_val = train.loc[val_idx, 'label'].to_numpy()
    lang_val = meta.detect_val_language()

    names, val_arrs, _ = meta.discover_preds()
    en_mask = (lang_val == 'en')
    de_mask = (lang_val == 'de')

    # ---- (1) per-language ensemble scores ----------------------------
    w_en, _ = meta.fit_weights_mae([p[en_mask] for p in val_arrs],
                                   y_val[en_mask], seed=42)
    w_de, _ = meta.fit_weights_mae([p[de_mask] for p in val_arrs],
                                   y_val[de_mask], seed=42)

    pred = np.empty(len(y_val), dtype=int)
    pred[en_mask] = median_round(
        meta.apply_weights([p[en_mask] for p in val_arrs], w_en))
    pred[de_mask] = median_round(
        meta.apply_weights([p[de_mask] for p in val_arrs], w_de))

    s_all = 1 - mean_absolute_error(y_val, pred) / 4
    s_en = 1 - mean_absolute_error(y_val[en_mask], pred[en_mask]) / 4
    s_de = 1 - mean_absolute_error(y_val[de_mask], pred[de_mask]) / 4
    print('--- Table 1, row "Weighted-MAE per-lang" ---')
    print(f'  val       = {s_all:.4f}   (report: 0.913)')
    print(f'  val_EN    = {s_en:.4f}   (report: 0.915)')
    print(f'  val_DE    = {s_de:.4f}   (report: 0.911)')

    # ---- (2) extreme confusions (0<->4) -------------------------------
    print('\n--- Section 4.5, extreme 0<->4 confusion rate ---')
    for lang, mask in [('EN', en_mask), ('DE', de_mask)]:
        cm = confusion_matrix(y_val[mask], pred[mask],
                              labels=[0, 1, 2, 3, 4], normalize='true')
        rate = (cm[0, 4] + cm[4, 0]) / 2
        print(f'  {lang}: P(pred=4|y=0) = {cm[0, 4]*100:.2f}%   '
              f'P(pred=0|y=4) = {cm[4, 0]*100:.2f}%   '
              f'avg = {rate*100:.2f}%   (report: < 0.5%)')

    # ---- (3) per-language Jaccard error overlap -----------------------
    print('\n--- Section 4.3, per-language Jaccard between models ---')
    for lang, mask in [('EN', en_mask), ('DE', de_mask)]:
        y = y_val[mask]
        errs = []
        for p in val_arrs:
            yhat = median_round(p[mask])
            errs.append(yhat != y)
        K = len(names)
        print(f'\n  val_{lang}  ({mask.sum()} reviews)')
        for i, ni in enumerate(names):
            row = []
            for j, nj in enumerate(names):
                inter = (errs[i] & errs[j]).sum()
                union = (errs[i] | errs[j]).sum()
                row.append(inter / max(union, 1))
            print('   ', ni.ljust(22), ' '.join(f'{v:5.3f}' for v in row))

    # ---- (4) the specific cell DeBERTa-EN x mDeBERTa on val_DE -------
    i = names.index('deberta_v3_large_en')
    j = names.index('mdeberta')
    err_i = (median_round(val_arrs[i][de_mask]) != y_val[de_mask])
    err_j = (median_round(val_arrs[j][de_mask]) != y_val[de_mask])
    jac = (err_i & err_j).sum() / max((err_i | err_j).sum(), 1)
    print(f'\nDeBERTa-EN x mDeBERTa on val_DE: Jaccard = {jac:.3f}   '
          f'(report: 0.60)')

    # ---- (5) 2-model uniform XLM-R + mDeBERTa (Section 4.1) -----------
    xi = names.index('xlmr_base')
    mi = names.index('mdeberta')
    mix = (val_arrs[xi] + val_arrs[mi]) / 2
    s_uni2 = 1 - mean_absolute_error(y_val, median_round(mix)) / 4
    print(f'\nXLM-R + mDeBERTa uniform ensemble val score: {s_uni2:.4f}   '
          f'(report: 0.908)')


if __name__ == '__main__':
    main()
