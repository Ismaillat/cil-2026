"""
B2 — Frozen multilingual sentence encoder + Logistic Regression.

Reproduces `b2_sentemb_logreg.ipynb` as a script. The encoder is
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-d).
Since `sentence-transformers` is not installed in the cluster CIL env,
we load the model with the plain `transformers` API and reimplement
the mean-pooling + L2 normalisation that `SentenceTransformer` does
internally; the resulting embeddings are numerically identical (up to
floating-point noise) to what the original notebook produced.

Outputs:
    preds/b2_val.npy   (25_200, 5)   softmax probabilities on val
    preds/b2_test.npy  (168_000, 5)  softmax probabilities on test
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MAX_LEN = 128
BATCH_SIZE = 256
SEED = 42
VAL_SIZE = 0.10


def median_round(p: np.ndarray) -> np.ndarray:
    return np.argmax(np.cumsum(p, axis=1) >= 0.5, axis=1)


def mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sentence-Transformers'-style mean pooling over non-padded tokens."""
    mask_f = mask.unsqueeze(-1).float()
    summed = (last_hidden * mask_f).sum(dim=1)
    counts = mask_f.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def encode(sentences, tokenizer, model, device, cache_path: str | None = None):
    if cache_path and os.path.exists(cache_path):
        print(f'  load cached embeddings from {cache_path}')
        return np.load(cache_path)
    model.eval()
    out = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = list(sentences[i:i + BATCH_SIZE])
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        h = model(**enc).last_hidden_state
        pooled = mean_pool(h, enc['attention_mask'])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        out.append(pooled.cpu().numpy())
        if (i // BATCH_SIZE) % 50 == 0:
            print(f'  encoded {i + len(batch):>7d} / {len(sentences)}', flush=True)
    emb = np.concatenate(out, axis=0)
    if cache_path:
        np.save(cache_path, emb)
        print(f'  cached embeddings to {cache_path}')
    return emb


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'{MODEL_NAME}  ({n_params:,} params)')

    os.makedirs('data', exist_ok=True)
    print('encoding train (~227k) ...', flush=True)
    X_train = encode(train_df['sentence'].tolist(), tokenizer, model, device,
                     cache_path='data/b2_train_emb.npy')
    print('encoding val (~25k) ...', flush=True)
    X_val = encode(val_df['sentence'].tolist(), tokenizer, model, device,
                   cache_path='data/b2_val_emb.npy')
    print('encoding test (~168k) ...', flush=True)
    X_test = encode(test['sentence'].tolist(), tokenizer, model, device,
                    cache_path='data/b2_test_emb.npy')

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print(f'X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}', flush=True)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    clf.fit(X_train, y_train)

    val_probs = clf.predict_proba(X_val)
    test_probs = clf.predict_proba(X_test)

    os.makedirs('preds', exist_ok=True)
    np.save('preds/b2_val.npy', val_probs)
    np.save('preds/b2_test.npy', test_probs)

    for tag, pred in [('argmax', np.argmax(val_probs, 1)),
                      ('median', median_round(val_probs))]:
        mae = mean_absolute_error(y_val, pred)
        print(f'[b2 {tag}] score={1 - mae / 4:.4f}  mae={mae:.4f}  '
              f'acc={(y_val == pred).mean():.4f}')


if __name__ == '__main__':
    main()
