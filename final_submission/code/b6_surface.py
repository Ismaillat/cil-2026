"""
B6 — Histogram Gradient Boosting on hand-crafted surface features.

A deliberately minimalist baseline whose only input signal is a small
set of *surface statistics* of the review text: lengths, punctuation
counts, capitalisation, repeated characters, emoji counts, and crude
language indicators.  No lexical content, no embeddings.  The model
therefore captures a signal that none of the other baselines uses
explicitly, maximising error-set diversity for the meta-learner.

Surface cues alone identify the level of sentiment intensity only
weakly (cf. the U-shape of ``!'' documented in the EDA), so the
standalone score is low; a meta-learner can still recover the
complementary error signal.

Output: preds/b6_surface_{val,test}.npy
"""

from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


SEED = 42
VAL_SIZE = 0.10

# ----- regular expressions for the surface features -----
_EXCL = re.compile(r'!')
_QST = re.compile(r'\?')
_DOT = re.compile(r'\.')
_COMMA = re.compile(r',')
_DIGIT = re.compile(r'\d')
_CAPS_WORD = re.compile(r'\b[A-ZÄÖÜ]{3,}\b')
_REPEAT_CHAR = re.compile(r'(.)\1{2,}')           # 3+ of the same character
_EMOJI = re.compile(r'[\U0001F300-\U0001FAFF\U00002600-\U000027BF]')
_STAR_BOLD = re.compile(r'\*[^*]+\*')             # *bolded text*
_DE_CHARS = set('äöüß')       # ä ö ü ß
_DE_FUNC = (' der ', ' die ', ' das ', ' und ', ' ich ', ' nicht ', ' ist ',
            ' ein ', ' eine ', ' mit ', ' für ', ' auch ', ' war ')
_EN_FUNC = (' the ', ' and ', ' is ', ' was ', ' have ', ' had ', ' but ',
            ' not ', ' very ', ' would ', ' could ')


def median_round(p: np.ndarray) -> np.ndarray:
    return np.argmax(np.cumsum(p, axis=1) >= 0.5, axis=1)


def features(text: str) -> np.ndarray:
    """Return a fixed-length numeric vector of surface features for one review."""
    s = text if isinstance(text, str) else ''
    n_chars = len(s) if s else 1
    n_words = max(len(s.split()), 1)
    lower = s.lower()
    padded_lower = f' {lower} '
    n_letters = sum(c.isalpha() for c in s)
    n_upper = sum(c.isupper() for c in s)
    return np.array([
        n_chars,                                          # 0
        n_words,                                          # 1
        n_chars / n_words,                                # 2  avg word length
        len(_EXCL.findall(s)),                            # 3
        len(_QST.findall(s)),                             # 4
        len(_DOT.findall(s)),                             # 5
        len(_COMMA.findall(s)),                           # 6
        len(_DIGIT.findall(s)),                           # 7
        len(_CAPS_WORD.findall(s)),                       # 8
        n_upper / max(n_letters, 1),                      # 9  uppercase ratio
        len(_REPEAT_CHAR.findall(s)),                     # 10 sooooo
        len(_EMOJI.findall(s)),                           # 11
        len(_STAR_BOLD.findall(s)),                       # 12 *bold*
        len(set(lower.split())) / n_words,                # 13 unique word ratio
        int(any(c in _DE_CHARS for c in s)),              # 14 has umlaut
        sum(w in padded_lower for w in _DE_FUNC),         # 15 DE function words
        sum(w in padded_lower for w in _EN_FUNC),         # 16 EN function words
        s.count('\n'),                                    # 17 line breaks
        s.count('  '),                                    # 18 double spaces
    ], dtype=np.float32)


def build_matrix(texts):
    return np.stack([features(t) for t in texts])


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

    print('extracting surface features ...', flush=True)
    X_train = build_matrix(train_df['sentence'].tolist())
    X_val = build_matrix(val_df['sentence'].tolist())
    X_test = build_matrix(test['sentence'].tolist())
    print(f'  shapes: train={X_train.shape}  val={X_val.shape}  test={X_test.shape}',
          flush=True)

    print('fitting HistGradientBoosting ...', flush=True)
    clf = HistGradientBoostingClassifier(
        max_iter=500, max_depth=6, learning_rate=0.05,
        l2_regularization=1.0, random_state=SEED, early_stopping=False,
    )
    clf.fit(X_train, y_train)
    print('  done', flush=True)

    val_probs = clf.predict_proba(X_val)
    test_probs = clf.predict_proba(X_test)

    os.makedirs('preds', exist_ok=True)
    np.save('preds/b6_surface_val.npy', val_probs)
    np.save('preds/b6_surface_test.npy', test_probs)

    for tag, pred in [('argmax', np.argmax(val_probs, 1)),
                      ('median', median_round(val_probs))]:
        mae = mean_absolute_error(y_val, pred)
        print(f'[b6_surface {tag}] score={1 - mae / 4:.4f}  mae={mae:.4f}  '
              f'acc={(y_val == pred).mean():.4f}', flush=True)


if __name__ == '__main__':
    main()
