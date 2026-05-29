"""
B3 — Kim-style CNN sentence classifier (wider variant) with regularisation.

A convolutional sentence-level classifier that complements the linear
(B1, B2) and recurrent (B4) baselines with a different inductive bias:
parallel n-gram detectors followed by max-over-time pooling.

  * vocabulary  : top 50,000 tokens of train
  * embedding   : learned 256-d
  * convolutions: kernel sizes (2, 3, 4, 5), 200 filters each
  * regularisation: BatchNorm after each conv, dropout 0.4, label smoothing 0.1
  * sequence    : max_length 256 tokens
  * optimisation: AdamW lr 5e-4, batch 64, 10 epochs, model selection on val score

~15M trainable parameters.  Drop-in compatible with the rest of the
pipeline:

    sbatch run.slurm train_cnn_big.py --name cnn_big

writes  preds/cnn_big_val.npy  and  preds/cnn_big_test.npy.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections import Counter
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_TOK_RE = re.compile(r"[\w']+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return _TOK_RE.findall(text.lower())


class Vocab:
    PAD = '<pad>'
    UNK = '<unk>'

    def __init__(self, itos: Sequence[str]):
        self.itos = list(itos)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.pad_idx = self.stoi[self.PAD]
        self.unk_idx = self.stoi[self.UNK]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Sequence[str], max_len: int) -> np.ndarray:
        ids = [self.stoi.get(t, self.unk_idx) for t in tokens[:max_len]]
        if len(ids) < max_len:
            ids += [self.pad_idx] * (max_len - len(ids))
        return np.asarray(ids, dtype=np.int64)

    @classmethod
    def build(cls, texts: Sequence[str], max_size: int) -> 'Vocab':
        counter: Counter = Counter()
        for t in texts:
            counter.update(tokenize(t))
        itos = [cls.PAD, cls.UNK] + [tok for tok, _ in counter.most_common(max_size)]
        return cls(itos)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ReviewDataset(Dataset):
    def __init__(self, texts: Sequence[str], vocab: Vocab, max_len: int,
                 labels: Sequence[int] | None = None):
        self.ids = np.stack([vocab.encode(tokenize(t), max_len) for t in texts])
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        item = {'input_ids': torch.from_numpy(self.ids[i])}
        if self.labels is not None:
            item['labels'] = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Model: wider Kim-2014 CNN with BatchNorm.
# ---------------------------------------------------------------------------

class TextCNNBig(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 num_filters: int = 200, filter_sizes=(2, 3, 4, 5),
                 num_classes: int = 5, dropout: float = 0.4,
                 pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=0)
            for k in filter_sizes
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_filters)
                                  for _ in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)                # (B, T, E)
        x = x.transpose(1, 2)                        # (B, E, T)
        pooled = []
        for conv, bn in zip(self.convs, self.bns):
            h = bn(conv(x))
            h = F.relu(h)
            h = F.max_pool1d(h, kernel_size=h.size(2)).squeeze(2)
            pooled.append(h)
        h = torch.cat(pooled, dim=1)
        h = self.dropout(h)
        return self.fc(h)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='cnn_big')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--max_length', type=int, default=256)
    ap.add_argument('--vocab_size', type=int, default=50000)
    ap.add_argument('--embed_dim', type=int, default=256)
    ap.add_argument('--num_filters', type=int, default=200)
    ap.add_argument('--dropout', type=float, default=0.4)
    ap.add_argument('--label_smoothing', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['input_ids'].to(device))
            out.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def median_round(p: np.ndarray) -> np.ndarray:
    return np.argmax(np.cumsum(p, axis=1) >= 0.5, axis=1)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}', flush=True)

    # ---- data ----
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    val_path = 'data/val_indices.npy'
    if os.path.exists(val_path):
        val_idx = np.load(val_path)
    else:
        _, val_idx = train_test_split(
            np.arange(len(train)), test_size=0.1,
            stratify=train['label'], random_state=42,
        )
        np.save(val_path, val_idx)
    train_df = train.drop(index=val_idx).reset_index(drop=True)
    val_df = train.loc[val_idx].reset_index(drop=True)
    y_val = val_df['label'].to_numpy()

    vocab = Vocab.build(train_df['sentence'].tolist(), max_size=args.vocab_size)
    print(f'vocab size: {len(vocab)}', flush=True)

    train_ds = ReviewDataset(train_df['sentence'].tolist(), vocab,
                             args.max_length, train_df['label'].tolist())
    val_ds = ReviewDataset(val_df['sentence'].tolist(), vocab,
                           args.max_length, val_df['label'].tolist())
    test_ds = ReviewDataset(test['sentence'].tolist(), vocab, args.max_length)

    nw = 2 if device.type == 'cuda' else 0
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=nw, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=nw)

    # ---- model / optim ----
    model = TextCNNBig(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        num_filters=args.num_filters,
        dropout=args.dropout,
        pad_idx=vocab.pad_idx,
    ).to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable params: {n_params:,}', flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ---- train loop with model selection on MAE-score ----
    best_score = -1.0
    best_val_probs = None
    best_test_probs = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_seen = 0
        for batch in train_loader:
            ids = batch['input_ids'].to(device, non_blocking=True)
            y = batch['labels'].to(device, non_blocking=True)
            logits = model(ids)
            loss = loss_fn(logits, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += loss.item() * y.size(0)
            n_seen += y.size(0)

        val_probs = predict_proba(model, val_loader, device)
        mae = mean_absolute_error(y_val, median_round(val_probs))
        score = 1 - mae / 4
        print(f'epoch {epoch:2d}  train_loss={running / n_seen:.4f}  '
              f'val_score={score:.4f}  val_mae={mae:.4f}  '
              f'time={time.time() - t0:.1f}s', flush=True)
        if score > best_score:
            best_score = score
            best_val_probs = val_probs
            best_test_probs = predict_proba(model, test_loader, device)
            print(f'  -> new best, regenerated test predictions', flush=True)

    os.makedirs('preds', exist_ok=True)
    np.save(f'preds/{args.name}_val.npy', best_val_probs)
    np.save(f'preds/{args.name}_test.npy', best_test_probs)

    for tag, pred in [('argmax', np.argmax(best_val_probs, 1)),
                      ('median', median_round(best_val_probs))]:
        mae = mean_absolute_error(y_val, pred)
        print(f'[{args.name} {tag}] score={1 - mae / 4:.4f}  mae={mae:.4f}  '
              f'acc={(y_val == pred).mean():.4f}', flush=True)


if __name__ == '__main__':
    main()
