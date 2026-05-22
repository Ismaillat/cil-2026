import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import mean_absolute_error


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--grad_accum', type=int, default=1)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--max_length', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def median_round(probs):
    return np.argmax(np.cumsum(probs, axis=1) >= 0.5, axis=1)


def main():
    args = parse_args()

    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    val_idx = np.load('data/val_indices.npy')
    train_df = train.drop(index=val_idx).reset_index(drop=True)
    val_df = train.loc[val_idx].reset_index(drop=True)
    y_val = val_df['label'].values

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    class ReviewDataset(Dataset):
        def __init__(self, texts, labels=None):
            self.texts = list(texts)
            self.labels = None if labels is None else list(labels)
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, i):
            enc = tokenizer(self.texts[i], truncation=True, max_length=args.max_length)
            item = {k: torch.tensor(v) for k, v in enc.items()}
            if self.labels is not None:
                item['labels'] = torch.tensor(int(self.labels[i]), dtype=torch.long)
            return item

    train_ds = ReviewDataset(train_df['sentence'], train_df['label'])
    val_ds = ReviewDataset(val_df['sentence'], val_df['label'])
    test_ds = ReviewDataset(test['sentence'])

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=5)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        mae = mean_absolute_error(labels, preds)
        return {'score': 1 - mae / 4, 'mae': mae}

    targs = TrainingArguments(
        output_dir=f'runs/{args.name}',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        fp16=torch.cuda.is_available(),
        max_grad_norm=1.0,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='score',
        greater_is_better=True,
        seed=args.seed,
        report_to=[],
        logging_steps=200,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    os.makedirs('preds', exist_ok=True)
    val_probs = softmax(trainer.predict(val_ds).predictions)
    test_probs = softmax(trainer.predict(test_ds).predictions)
    np.save(f'preds/{args.name}_val.npy', val_probs)
    np.save(f'preds/{args.name}_test.npy', test_probs)

    for tag, pred in [('argmax', np.argmax(val_probs, 1)),
                      ('median', median_round(val_probs))]:
        mae = mean_absolute_error(y_val, pred)
        print(f'[{args.name} {tag}] score={1 - mae / 4:.4f} mae={mae:.4f}')


if __name__ == '__main__':
    main()
