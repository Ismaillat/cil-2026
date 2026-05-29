"""Bootstrap a 95% confidence interval on the per-language ensemble.

Reproduces the sentence

    "bootstrap resampling on the ensemble gives a 95% CI of
    [0.9100, 0.9143], comfortably above 0.906."

(Section 5 of the report). The ensemble is the per-language
weighted-MAE meta-learner: for every validation sample take the
mixture defined by w_en or w_de depending on the detected language,
decode with the median rule, then resample the 25 200 per-review
errors with replacement N_BOOT times and read off the 2.5/97.5
percentiles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import meta  # reuse helpers + the simplex search


N_BOOT = 10_000
SEED = 42


def main():
    rng = np.random.default_rng(SEED)

    train = pd.read_csv('data/train.csv')
    val_idx = np.load('data/val_indices.npy')
    y_val = train.loc[val_idx, 'label'].to_numpy()
    lang_val = meta.detect_val_language()

    names, val_arrs, _ = meta.discover_preds()
    print(f'pool ({len(names)} models): {names}')

    # Refit the per-language simplex on the full validation set.
    en_mask = (lang_val == 'en')
    de_mask = (lang_val == 'de')

    w_en, _ = meta.fit_weights_mae([p[en_mask] for p in val_arrs],
                                   y_val[en_mask], seed=SEED)
    w_de, _ = meta.fit_weights_mae([p[de_mask] for p in val_arrs],
                                   y_val[de_mask], seed=SEED)

    mix_en = meta.apply_weights([p[en_mask] for p in val_arrs], w_en)
    mix_de = meta.apply_weights([p[de_mask] for p in val_arrs], w_de)

    pred = np.empty(len(y_val), dtype=int)
    pred[en_mask] = meta.median_round(mix_en)
    pred[de_mask] = meta.median_round(mix_de)

    point = 1 - mean_absolute_error(y_val, pred) / 4
    print(f'point estimate (val score)        = {point:.4f}')

    errs = np.abs(y_val - pred)              # per-sample MAE contribution
    n = len(errs)
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boots[b] = 1 - errs[idx].mean() / 4

    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f'95% CI ({N_BOOT}-resample bootstrap) = [{lo:.4f}, {hi:.4f}]')
    print(f'distance to 0.906 baseline:       '
          f'point {point - 0.906:+.4f}   '
          f'CI-low {lo - 0.906:+.4f}')


if __name__ == '__main__':
    main()
