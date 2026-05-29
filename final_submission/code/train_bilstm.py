"""
B4 — Bidirectional LSTM sentence classifier.

A recurrent neural baseline that complements the CNN family (B3) with a
different inductive bias: sequential processing with long-range memory
rather than local convolutional features.

Architecture:
  * vocabulary  : top 50,000 tokens of train
  * embedding   : learned 256-d
  * encoder     : 2-layer bidirectional LSTM with hidden 256 per direction
  * pooling     : concatenation of max-pool and last-step features over time
  * head        : dropout 0.4 -> linear -> 5 classes
  * regularisation: label smoothing 0.1, weight decay 1e-4

Run as

    python train_bilstm.py --name bilstm

writes preds/bilstm_val.npy and preds/bilstm_test.npy.
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
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_idx = self.stoi[self.PAD]
        self.unk_idx = self.stoi[self.UNK]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Sequence[str], max_len: int) -> np.ndarray:
        ids = [self.stoi.get(t, self.unk_idx) for t in tokens[:max_len]]
        n = len(ids)
        if n < max_len:
            ids += [self.pad_idx] * (max_len - n)
        return np.asarray(ids, dtype=np.int64), n

    @classmethod
    def build(cls, texts: Sequence[str], max_size: int) -> 'Vocab':
        counter: Counter = Counter()
        for t in texts:
            counter.update(tokenize(t))
        itos = [cls.PAD, cls.UNK] + [tok for tok, _ in counter.most_common(max_size)]
        return cls(itos)


class ReviewDataset(Dataset):
    def __init__(self, texts: Sequence[str], vocab: Vocab, max_len: int,
                 labels: Sequence[int] | None = None):
        ids, lens = [], []
        for t in texts:
            i, n = vocab.encode(tokenize(t), max_len)
            ids.append(i)
            lens.append(max(1, n))
        self.ids = np.stack(ids)
        self.lens = np.asarray(lens, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        item = {'input_ids': torch.from_numpy(self.ids[i]),
                'lengths': torch.tensor(int(self.lens[i]), dtype=torch.long)}
        if self.labels is not None:
            item['labels'] = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return item


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 hidden_dim: int = 256, num_layers: int = 2,
                 num_classes: int = 5, dropout: float = 0.4, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # Pool: concat(max-over-time, last forward + last backward) = 4*hidden
        self.fc = nn.Linear(4 * hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)                          # (B, T, E)
        # Pack to ignore padding.
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        out_p, (h_n, _) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_p, batch_first=True)
        # Max-over-time pooling, ignoring padded positions.
        T = out.size(1)
        mask = (torch.arange(T, device=out.device)
                .unsqueeze(0)
                < lengths.to(out.device).unsqueeze(1)).unsqueeze(-1)
        out_masked = out.masked_fill(~mask, float('-inf'))
        max_pooled = out_masked.max(dim=1).values
        # Last-step features: last layer's forward + backward hidden.
        last_fwd = h_n[-2]
        last_bwd = h_n[-1]
        feat = torch.cat([max_pooled, last_fwd, last_bwd], dim=1)
        feat = self.dropout(feat)
        return self.fc(feat)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='bilstm')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--max_length', type=int, default=200)
    ap.add_argument('--vocab_size', type=int, default=50000)
    ap.add_argument('--embed_dim', type=int, default=256)
    ap.add_argument('--hidden_dim', type=int, default=256)
    ap.add_argument('--num_layers', type=int, default=2)
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
            logits = model(batch['input_ids'].to(device),
                           batch['lengths'].to(device))
            out.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def median_round(p: np.ndarray) -> np.ndarray:
    return np.argmax(np.cumsum(p, axis=1) >= 0.5, axis=1)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}', flush=True)

    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    val_path = 'data/val_indices.npy'
    if os.path.exists(val_path):
        val_idx = np.load(val_path)
    else:
        _, val_idx = train_test_split(np.arange(len(train)), test_size=0.1,
                                      stratify=train['label'], random_state=42)
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
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=nw)

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=vocab.pad_idx,
    ).to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable params: {n_params:,}', flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

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
            lens = batch['lengths'].to(device, non_blocking=True)
            y = batch['labels'].to(device, non_blocking=True)
            logits = model(ids, lens)
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
            print('  -> new best, regenerated test predictions', flush=True)

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
