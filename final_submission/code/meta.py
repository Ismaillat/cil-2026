"""Meta-learner over the per-model softmax probabilities.

Six aggregation strategies are compared, all decoded with the
MAE-optimal median rule (Proposition 3.1 of the report):

  uniform                 Uniform average of probabilities.
  weighted_mae            One scalar weight per model on the simplex,
                          optimised by Dirichlet sampling + coordinate
                          refinement to minimise MAE after median
                          decoding on the validation hold-out.
  weighted_mae_per_lang   Two simplex weight vectors (w_en, w_de) fit
                          separately on the EN/DE partitions of val;
                          test samples are routed by detected language.
  logreg                  Multinomial logistic regression on the 5K
                          concatenated probability features.
  gbm                     Histogram gradient boosting on the same.
  gating_mlp              Soft per-sample routing by a small MLP whose
                          input is the MiniLM embedding + per-model
                          uncertainty signals + language flag.  Trained
                          with a Wasserstein-1 (EMD) ordinal loss.

Every variant is scored by 5-fold stratified cross-validation on the
validation hold-out; the OOF score is the only number we use to choose
between methods.  For the test submission we re-fit each variant on the
full val hold-out and write one submissions/meta_*.csv per variant.

Run:
    python meta.py
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def median_round(p: np.ndarray) -> np.ndarray:
    """Bayes-optimal integer predictor under absolute error on ordinal labels."""
    return np.argmax(np.cumsum(p, axis=1) >= 0.5, axis=1)


def argmax(p: np.ndarray) -> np.ndarray:
    return np.argmax(p, axis=1)


def score_of(y_true: np.ndarray, y_pred: np.ndarray):
    mae = mean_absolute_error(y_true, y_pred)
    return 1 - mae / 4, mae, float((y_true == y_pred).mean())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_preds(pred_dir: str = 'preds'):
    """Return (names, val_arrays, test_arrays) for every model with both files."""
    val_files = sorted(glob.glob(os.path.join(pred_dir, '*_val.npy')))
    names, val_arrs, test_arrs = [], [], []
    for vf in val_files:
        tf = vf.replace('_val.npy', '_test.npy')
        if not os.path.exists(tf):
            continue
        names.append(os.path.basename(vf).replace('_val.npy', ''))
        val_arrs.append(np.load(vf).astype(np.float64))
        test_arrs.append(np.load(tf).astype(np.float64))
    return names, val_arrs, test_arrs


def load_val_labels() -> np.ndarray:
    train = pd.read_csv('data/train.csv')
    val_idx = np.load('data/val_indices.npy')
    return train.loc[val_idx, 'label'].to_numpy()


# Same German-detection heuristic as in split_by_language.py.
_DE_CHARS = ('ä', 'ö', 'ü', 'ß')
_DE_WORDS = (' der ', ' die ', ' das ', ' und ', ' ich ', ' nicht ', ' ist ',
             ' ein ', ' eine ', ' mit ', ' für ', ' auch ', ' war ',
             ' es ', ' zu ', ' den ')


def _is_de(text: str) -> bool:
    t = (text or '').lower()
    if any(c in t for c in _DE_CHARS):
        return True
    padded = f' {t} '
    return any(w in padded for w in _DE_WORDS)


def detect_language(csv_path: str) -> np.ndarray:
    """Return an array of 'en'/'de' labels for each row of csv_path."""
    sentences = pd.read_csv(csv_path)['sentence'].tolist()
    return np.array(['de' if _is_de(s) else 'en' for s in sentences])


def detect_val_language() -> np.ndarray:
    train = pd.read_csv('data/train.csv')
    val_idx = np.load('data/val_indices.npy')
    sentences = train.loc[val_idx, 'sentence'].tolist()
    return np.array(['de' if _is_de(s) else 'en' for s in sentences])


def stack_features(arrs: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(arrs, axis=1)


# ---------------------------------------------------------------------------
# Combination 1: weighted average with MAE-direct optimisation
# ---------------------------------------------------------------------------

def fit_weights_mae(preds: List[np.ndarray], y: np.ndarray,
                    n_samples: int = 8000, n_refine: int = 3,
                    seed: int = 42):
    """Search w on the K-simplex to minimise MAE after median decoding."""
    K = len(preds)
    P = np.stack(preds, axis=0)             # (K, n, 5)
    rng = np.random.default_rng(seed)

    # Stage 1: random simplex search.
    samples = rng.dirichlet(np.ones(K), size=n_samples)
    samples = np.vstack([samples, np.eye(K)])  # also try each model alone
    best_mae = np.inf
    best_w = np.ones(K) / K
    for w in samples:
        mix = np.tensordot(w, P, axes=([0], [0]))     # (n, 5)
        mae = mean_absolute_error(y, median_round(mix))
        if mae < best_mae:
            best_mae = mae
            best_w = w

    # Stage 2: coordinate refinement.
    deltas = np.linspace(-0.30, 0.30, 31)
    for _ in range(n_refine):
        improved = False
        for i in range(K):
            for d in deltas:
                cand = best_w.copy()
                cand[i] = max(0.0, cand[i] + d)
                s = cand.sum()
                if s <= 0:
                    continue
                cand = cand / s
                mix = np.tensordot(cand, P, axes=([0], [0]))
                mae = mean_absolute_error(y, median_round(mix))
                if mae + 1e-9 < best_mae:
                    best_mae = mae
                    best_w = cand
                    improved = True
        if not improved:
            break

    return best_w, best_mae


def apply_weights(preds: List[np.ndarray], w: np.ndarray) -> np.ndarray:
    P = np.stack(preds, axis=0)
    return np.tensordot(w, P, axes=([0], [0]))


# ---------------------------------------------------------------------------
# Combination 2 / 3: classical stacking on concatenated probabilities
# ---------------------------------------------------------------------------

def make_classifier(name: str, seed: int):
    if name == 'logreg':
        return LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs',
                                  random_state=seed)
    if name == 'gbm':
        return HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=400,
            l2_regularization=1.0, random_state=seed,
        )
    raise ValueError(name)


def predict_5(clf, X: np.ndarray) -> np.ndarray:
    """Predict (N, 5) probabilities even if classifier never saw a class."""
    p = clf.predict_proba(X)
    if p.shape[1] == 5 and (clf.classes_ == np.arange(5)).all():
        return p
    out = np.zeros((X.shape[0], 5))
    for i, c in enumerate(clf.classes_):
        out[:, int(c)] = p[:, i]
    return out


# ---------------------------------------------------------------------------
# Combination 4: gating MLP -- per-sample weights from sentence embedding +
# per-model uncertainty signals.  Trained with an ordinal Wasserstein-1
# loss (matches MAE on integer labels) and averaged over multiple seeds.
# ---------------------------------------------------------------------------

class GatingMLP(nn.Module):
    """Predicts softmax weights over K models from per-sample features."""

    def __init__(self, in_dim: int, K: int, hidden: int = 64,
                 dropout: float = 0.5, input_dropout: float = 0.1):
        super().__init__()
        self.input_dropout = nn.Dropout(input_dropout)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, K),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(self.input_dropout(x)), dim=-1)


def gating_mix(probs_stack: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """probs_stack: (B, K, 5), weights: (B, K) -> mixed (B, 5)."""
    return (probs_stack * weights.unsqueeze(-1)).sum(dim=1)


def gating_features(emb: np.ndarray, preds: List[np.ndarray],
                    lang: np.ndarray | None = None) -> np.ndarray:
    """Concat the sentence embedding with per-model uncertainty signals
    (and, optionally, a binary language flag).

    Final feature vector per sample:
      - emb           (E)     :  semantic content (MiniLM)
      - max-prob      (K)     :  each model's confidence on its top class
      - entropy       (K)     :  each model's predictive entropy
      - is_de         (1)     :  language flag (DE=1, EN=0), optional
    Lets the gating network condition on "who is confident on what",
    and -- when the language flag is included -- on whether to route
    the EN-specialist or the DE-specialist.
    """
    K = len(preds)
    max_probs = np.stack([p.max(axis=1) for p in preds], axis=1)            # (n, K)
    entropies = -np.stack([
        (p * np.log(np.clip(p, 1e-9, 1.0))).sum(axis=1) for p in preds
    ], axis=1)                                                              # (n, K)
    parts = [emb, max_probs, entropies]
    if lang is not None:
        parts.append((lang == 'de').astype(np.float32).reshape(-1, 1))
    return np.concatenate(parts, axis=1).astype(np.float32)


def wasserstein1_loss(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Ordinal Wasserstein-1 / EMD loss between predicted distribution p and
    one-hot label y on integer classes {0,...,C-1}.  Equals expected MAE
    under the predicted distribution and is differentiable in p, so it is
    the correct surrogate for our MAE-based metric."""
    C = p.shape[-1]
    cdf_p = torch.cumsum(p, dim=-1)                                         # (B, C)
    ks = torch.arange(C, device=p.device).unsqueeze(0)                      # (1, C)
    cdf_y = (ks >= y.unsqueeze(1)).float()                                  # (B, C)
    return (cdf_p - cdf_y).abs().sum(dim=-1).mean()


def fit_gating(features: np.ndarray, preds: List[np.ndarray], y: np.ndarray,
               *, hidden: int = 64, dropout: float = 0.5,
               input_dropout: float = 0.1, weight_decay: float = 1e-3,
               lr: float = 1e-3, epochs: int = 15, batch_size: int = 256,
               device: torch.device | None = None, seed: int = 42):
    device = device or torch.device('cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)

    K = len(preds)
    X = torch.tensor(features, dtype=torch.float32, device=device)
    P = torch.tensor(np.stack(preds, axis=1), dtype=torch.float32,
                     device=device)
    Y = torch.tensor(y, dtype=torch.long, device=device)

    model = GatingMLP(features.shape[1], K, hidden=hidden, dropout=dropout,
                      input_dropout=input_dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr,
                              weight_decay=weight_decay)

    n = X.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            w = model(X[idx])
            mix = gating_mix(P[idx], w)
            loss = wasserstein1_loss(mix, Y[idx])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

    return model


@torch.no_grad()
def gating_predict(model: nn.Module, features: np.ndarray,
                   preds: List[np.ndarray], device: torch.device,
                   batch_size: int = 2048) -> np.ndarray:
    model.eval()
    X = torch.tensor(features, dtype=torch.float32, device=device)
    P = torch.tensor(np.stack(preds, axis=1), dtype=torch.float32,
                     device=device)
    n = X.shape[0]
    out = np.zeros((n, 5), dtype=np.float64)
    for s in range(0, n, batch_size):
        e = s + batch_size
        w = model(X[s:e])
        mix = gating_mix(P[s:e], w)
        out[s:e] = mix.cpu().numpy()
    return out


def fit_predict_gating_multiseed(features_train: np.ndarray,
                                 preds_train: List[np.ndarray],
                                 y_train: np.ndarray,
                                 features_pred: np.ndarray,
                                 preds_pred: List[np.ndarray],
                                 *, n_seeds: int = 3, base_seed: int = 42,
                                 device: torch.device | None = None) -> np.ndarray:
    """Train n_seeds gating MLPs and average their mixed posteriors."""
    device = device or torch.device('cpu')
    acc = None
    for s in range(n_seeds):
        model = fit_gating(features_train, preds_train, y_train,
                           device=device, seed=base_seed + s)
        mix = gating_predict(model, features_pred, preds_pred, device)
        acc = mix if acc is None else acc + mix
    return acc / n_seeds


# ---------------------------------------------------------------------------
# Cross-validated evaluation
# ---------------------------------------------------------------------------

def kfold_oof_weighted_mae_per_lang(preds_val, preds_test, y_val,
                                    lang_val, lang_test, folds, seed):
    """Per-language weighted_mae: fit separate simplex weights on EN/DE val
    folds, then route each test sample to the matching weight vector.

    For each fold, EN and DE samples within the fold's training portion are
    used to fit `w_en` and `w_de` independently; OOF predictions are then
    produced on the held-out portion using the language-specific weights.
    Final submission re-fits on the full val_en / val_de.
    """
    n = y_val.shape[0]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_pred = np.zeros(n, dtype=int)
    fold_scores = []
    fold_weights_en, fold_weights_de = [], []

    for tr, te in skf.split(np.arange(n), y_val):
        tr_en = tr[lang_val[tr] == 'en']
        tr_de = tr[lang_val[tr] == 'de']
        w_en, _ = fit_weights_mae([p[tr_en] for p in preds_val],
                                  y_val[tr_en], seed=seed)
        w_de, _ = fit_weights_mae([p[tr_de] for p in preds_val],
                                  y_val[tr_de], seed=seed)
        fold_weights_en.append(w_en)
        fold_weights_de.append(w_de)

        te_en = te[lang_val[te] == 'en']
        te_de = te[lang_val[te] == 'de']
        if len(te_en):
            mix = apply_weights([p[te_en] for p in preds_val], w_en)
            oof_pred[te_en] = median_round(mix)
        if len(te_de):
            mix = apply_weights([p[te_de] for p in preds_val], w_de)
            oof_pred[te_de] = median_round(mix)
        fold_scores.append(1 - mean_absolute_error(y_val[te], oof_pred[te]) / 4)

    # Final fit on FULL val_en and val_de.
    en_mask = lang_val == 'en'
    de_mask = lang_val == 'de'
    w_en_final, _ = fit_weights_mae([p[en_mask] for p in preds_val],
                                    y_val[en_mask], seed=seed)
    w_de_final, _ = fit_weights_mae([p[de_mask] for p in preds_val],
                                    y_val[de_mask], seed=seed)

    # Apply per-language weights on the test set.
    en_test = lang_test == 'en'
    de_test = lang_test == 'de'
    test_pred = np.zeros(lang_test.shape[0], dtype=int)
    if en_test.any():
        mix = apply_weights([p[en_test] for p in preds_test], w_en_final)
        test_pred[en_test] = median_round(mix)
    if de_test.any():
        mix = apply_weights([p[de_test] for p in preds_test], w_de_final)
        test_pred[de_test] = median_round(mix)

    return oof_pred, fold_scores, (
        np.array(fold_weights_en), np.array(fold_weights_de),
        w_en_final, w_de_final), test_pred


def kfold_oof_weighted_mae(preds_val, preds_test, y_val, folds, seed):
    """Per-fold weight refit, OOF integer predictions + test refit."""
    n = y_val.shape[0]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_pred = np.zeros(n, dtype=int)
    fold_weights = []
    fold_scores = []
    for tr, te in skf.split(np.arange(n), y_val):
        w, _ = fit_weights_mae([p[tr] for p in preds_val], y_val[tr],
                               seed=seed)
        fold_weights.append(w)
        mix_te = apply_weights([p[te] for p in preds_val], w)
        oof_pred[te] = median_round(mix_te)
        fold_scores.append(1 - mean_absolute_error(y_val[te], oof_pred[te]) / 4)
    fold_weights = np.array(fold_weights)
    # Final refit on full val for the actual submission.
    w_final, _ = fit_weights_mae(preds_val, y_val, seed=seed)
    test_pred = median_round(apply_weights(preds_test, w_final))
    return oof_pred, fold_scores, fold_weights, w_final, test_pred


def kfold_oof_gating(preds_val, preds_test, y_val, emb_val, emb_test,
                     folds, seed, n_seeds: int = 3,
                     lang_val: np.ndarray | None = None,
                     lang_test: np.ndarray | None = None):
    """Per-fold multi-seed gating-MLP refit; OOF + final-refit predictions.

    Each fold trains `n_seeds` gating MLPs from different inits and averages
    their mixed posteriors, then decodes with the median.  The features fed
    to the gating include the sentence embedding *plus* per-model max-prob
    and per-model predictive entropy, so the gating can directly react to
    uncertainty and inter-model disagreement.
    """
    n = y_val.shape[0]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          ('mps' if torch.backends.mps.is_available()
                           else 'cpu'))

    feats_val = gating_features(emb_val, preds_val, lang=lang_val)
    feats_test = gating_features(emb_test, preds_test, lang=lang_test)

    oof_pred = np.zeros(n, dtype=int)
    fold_scores = []
    for tr, te in skf.split(np.arange(n), y_val):
        mix_te = fit_predict_gating_multiseed(
            feats_val[tr], [p[tr] for p in preds_val], y_val[tr],
            feats_val[te], [p[te] for p in preds_val],
            n_seeds=n_seeds, base_seed=seed, device=device,
        )
        oof_pred[te] = median_round(mix_te)
        fold_scores.append(
            1 - mean_absolute_error(y_val[te], oof_pred[te]) / 4
        )

    # Final: refit on full val, predict on test (also seed-averaged).
    mix_test = fit_predict_gating_multiseed(
        feats_val, preds_val, y_val, feats_test, preds_test,
        n_seeds=n_seeds, base_seed=seed, device=device,
    )
    test_pred = median_round(mix_test)
    return oof_pred, fold_scores, None, test_pred


def kfold_oof_classical(preds_val, preds_test, y_val, kind, folds, seed):
    """Per-fold refit of a classical stacker, OOF probabilities + test refit."""
    n = y_val.shape[0]
    X_val = stack_features(preds_val)
    X_test = stack_features(preds_test)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_proba = np.zeros((n, 5))
    fold_scores = []
    for fold, (tr, te) in enumerate(skf.split(X_val, y_val), 1):
        clf = make_classifier(kind, seed=seed + fold)
        clf.fit(X_val[tr], y_val[tr])
        oof_proba[te] = predict_5(clf, X_val[te])
        fold_scores.append(
            1 - mean_absolute_error(y_val[te],
                                    median_round(oof_proba[te])) / 4
        )
    clf_final = make_classifier(kind, seed=seed)
    clf_final.fit(X_val, y_val)
    test_proba = predict_5(clf_final, X_test)
    test_pred = median_round(test_proba)
    return median_round(oof_proba), fold_scores, clf_final, test_pred


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_score(label: str, y_true: np.ndarray, y_pred: np.ndarray):
    s, m, a = score_of(y_true, y_pred)
    print(f'  {label:38s}  score={s:.4f}  mae={m:.4f}  acc={a:.4f}')
    return s


def hr(t: str):
    print('\n' + '=' * 72 + f'\n{t}\n' + '=' * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--submission', default='meta_weighted_mae.csv',
                    help='which method writes the final submission')
    return ap.parse_args()


def main():
    args = parse_args()

    names, val_arrs, test_arrs = discover_preds()
    if not names:
        raise SystemExit('no preds/*_val.npy found')
    print(f'models found ({len(names)}): {names}')

    y_val = load_val_labels()
    lang_val = detect_val_language()
    lang_test = detect_language('data/test.csv')
    print(f'val composition: en={(lang_val=="en").sum()}  de={(lang_val=="de").sum()}')
    print(f'test composition: en={(lang_test=="en").sum()}  de={(lang_test=="de").sum()}')

    hr('[1] Per-model + uniform average (sanity baselines)')
    for name, p in zip(names, val_arrs):
        print_score(f'{name} / median', y_val, median_round(p))
    uniform_val = sum(val_arrs) / len(val_arrs)
    uniform_test = sum(test_arrs) / len(test_arrs)
    uniform_score = print_score('UNIFORM AVG / median', y_val,
                                median_round(uniform_val))

    hr('[2] Weighted MAE-direct (per-model simplex weights)')
    oof_pred, fold_scores, fold_weights, w_final, test_pred_w = \
        kfold_oof_weighted_mae(val_arrs, test_arrs, y_val,
                               folds=args.folds, seed=args.seed)
    print(f'  per-fold scores: {[round(s, 4) for s in fold_scores]}  '
          f'std={np.std(fold_scores):.4f}')
    weighted_oof_score = print_score('weighted_mae (OOF)', y_val, oof_pred)
    print('  per-fold weights:')
    for i, name in enumerate(names):
        col = fold_weights[:, i]
        print(f'    {name:14s}  ' + '  '.join(f'{c:.3f}' for c in col) +
              f'   mean={col.mean():.3f}±{col.std():.3f}')
    print('  full-val weights (used for test submission):')
    for name, w in zip(names, w_final):
        print(f'    {name:14s}  {w:.3f}')

    hr('[2b] Weighted MAE per-language (specialist routing)')
    oof_perlang, fold_perlang, perlang_weights, test_pred_perlang = \
        kfold_oof_weighted_mae_per_lang(val_arrs, test_arrs, y_val,
                                        lang_val, lang_test,
                                        folds=args.folds, seed=args.seed)
    fw_en, fw_de, w_en_final, w_de_final = perlang_weights
    print(f'  per-fold scores: {[round(s,4) for s in fold_perlang]}  '
          f'std={np.std(fold_perlang):.4f}')
    perlang_oof_score = print_score('weighted_mae_per_lang (OOF)', y_val, oof_perlang)
    print('  final-fit weights (used for the test submission):')
    for i, name in enumerate(names):
        print(f'    {name:20s}  w_en={w_en_final[i]:.3f}   w_de={w_de_final[i]:.3f}')

    hr('[3] Stacked LogReg (log-loss objective) on concatenated probas')
    oof_lr, fold_lr, _, test_pred_lr = kfold_oof_classical(
        val_arrs, test_arrs, y_val, 'logreg', args.folds, args.seed)
    print(f'  per-fold scores: {[round(s, 4) for s in fold_lr]}  '
          f'std={np.std(fold_lr):.4f}')
    lr_oof_score = print_score('logreg (OOF)', y_val, oof_lr)

    hr('[4] Stacked GBM (HistGradientBoosting) on concatenated probas')
    oof_gbm, fold_gbm, _, test_pred_gbm = kfold_oof_classical(
        val_arrs, test_arrs, y_val, 'gbm', args.folds, args.seed)
    print(f'  per-fold scores: {[round(s, 4) for s in fold_gbm]}  '
          f'std={np.std(fold_gbm):.4f}')
    gbm_oof_score = print_score('gbm (OOF)', y_val, oof_gbm)

    gating_score = None
    test_pred_gating = None
    emb_val_path = 'data/b2_val_emb.npy'
    emb_test_path = 'data/b2_test_emb.npy'
    if os.path.exists(emb_val_path) and os.path.exists(emb_test_path):
        hr('[5] Gating MLP (per-sample weights from B2 sentence embedding)')
        emb_val = np.load(emb_val_path).astype(np.float32)
        emb_test = np.load(emb_test_path).astype(np.float32)
        oof_gating, fold_gating, _, test_pred_gating = kfold_oof_gating(
            val_arrs, test_arrs, y_val, emb_val, emb_test,
            folds=args.folds, seed=args.seed,
            lang_val=lang_val, lang_test=lang_test)
        print(f'  per-fold scores: {[round(s, 4) for s in fold_gating]}  '
              f'std={np.std(fold_gating):.4f}')
        gating_score = print_score('gating_mlp (OOF)', y_val, oof_gating)
    else:
        print('\n[5] Gating MLP skipped: data/b2_{val,test}_emb.npy missing')

    hr('[6] Summary')
    print(f'  uniform_avg                = {uniform_score:.4f}  (current method)')
    print(f'  weighted_mae OOF           = {weighted_oof_score:.4f}'
          f'  ({weighted_oof_score - uniform_score:+.4f})')
    print(f'  weighted_mae_per_lang OOF  = {perlang_oof_score:.4f}'
          f'  ({perlang_oof_score - uniform_score:+.4f})  *language-routed*')
    print(f'  logreg OOF                 = {lr_oof_score:.4f}'
          f'  ({lr_oof_score - uniform_score:+.4f})')
    print(f'  gbm OOF                    = {gbm_oof_score:.4f}'
          f'  ({gbm_oof_score - uniform_score:+.4f})')
    if gating_score is not None:
        print(f'  gating_mlp OOF    = {gating_score:.4f}'
              f'  ({gating_score - uniform_score:+.4f})')

    # ---- write the chosen submission ----
    test_map = {
        'meta_weighted_mae.csv':          test_pred_w,
        'meta_weighted_mae_per_lang.csv': test_pred_perlang,
        'meta_logreg.csv':                test_pred_lr,
        'meta_gbm.csv':                   test_pred_gbm,
    }
    if test_pred_gating is not None:
        test_map['meta_gating.csv'] = test_pred_gating

    # Write ALL submissions (not just the chosen one) so we can compare on Kaggle.
    test_ids = pd.read_csv('data/test.csv')['id'].to_numpy()
    os.makedirs('submissions', exist_ok=True)
    for fname, pred in test_map.items():
        pd.DataFrame({'id': test_ids, 'label': pred.astype(int)}).to_csv(
            os.path.join('submissions', fname), index=False)
        print(f'  wrote submissions/{fname}')
    # Default submission of the script (also written above).
    if args.submission in test_map:
        pred = test_map[args.submission]
        print(f'\n  default submission = {args.submission}  '
              f'(label dist: {dict(zip(*np.unique(pred, return_counts=True)))})')


if __name__ == '__main__':
    main()
