"""Generate report figures and statistics from preds/ and data/."""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, mean_absolute_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, 'report', 'figs')
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 200,
})


def median_round(probs):
    return np.argmax(np.cumsum(probs, axis=1) >= 0.5, axis=1)


train = pd.read_csv(os.path.join(ROOT, 'data', 'train.csv'))
test = pd.read_csv(os.path.join(ROOT, 'data', 'test.csv'))
val_idx = np.load(os.path.join(ROOT, 'data', 'val_indices.npy'))
val_df = train.loc[val_idx].reset_index(drop=True)
y_val = val_df['label'].values

xlmr_val = np.load(os.path.join(ROOT, 'preds', 'xlmr_base_val.npy'))
mdeb_val = np.load(os.path.join(ROOT, 'preds', 'mdeberta_val.npy'))
ens_val = (xlmr_val + mdeb_val) / 2

xlmr_test = np.load(os.path.join(ROOT, 'preds', 'xlmr_base_test.npy'))
mdeb_test = np.load(os.path.join(ROOT, 'preds', 'mdeberta_test.npy'))


# ---------- FIGURE 1: surface features by label ----------
n_excl = train['sentence'].str.count('!')
n_qst = train['sentence'].str.count(r'\?')
m_excl = n_excl.groupby(train['label']).mean()
m_qst = n_qst.groupby(train['label']).mean()

fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.6))
labels5 = ['1', '2', '3', '4', '5']
axes[0].bar(labels5, m_excl.values, color='#3b6fb5')
axes[0].set_title("Mean count of '!'")
axes[0].set_xlabel('star rating')
axes[0].set_ylim(0, max(m_excl.max(), 0.8) * 1.1)

axes[1].bar(labels5, m_qst.values, color='#3b6fb5')
axes[1].set_title("Mean count of '?'")
axes[1].set_xlabel('star rating')
axes[1].set_ylim(0, max(m_qst.max(), 0.1) * 1.15)

fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'surface_features.pdf'), bbox_inches='tight')
plt.close(fig)
print('wrote surface_features.pdf')


# ---------- FIGURE 2: length distribution ----------
nw = train['sentence'].str.split().str.len()
fig, ax = plt.subplots(figsize=(3.3, 1.7))
ax.hist(nw.clip(upper=300), bins=50, color='#3b6fb5', edgecolor='white', linewidth=0.3)
p99 = int(np.percentile(nw, 99))
ax.axvline(p99, color='#c4302b', linestyle='--', linewidth=1.0,
           label=f'99th pct = {p99}')
ax.axvline(256, color='black', linestyle=':', linewidth=1.0,
           label='max_length = 256')
ax.set_xlabel('words per review (clipped at 300)')
ax.set_ylabel('reviews')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'length_hist.pdf'), bbox_inches='tight')
plt.close(fig)
print('wrote length_hist.pdf')


# ---------- FIGURE 3: ensemble confusion matrix ----------
pred_ens = median_round(ens_val)
cm = confusion_matrix(y_val, pred_ens, labels=[0, 1, 2, 3, 4])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(2.8, 2.7))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(labels5)
ax.set_yticklabels(labels5)
ax.set_xlabel('predicted')
ax.set_ylabel('true')
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{cm_norm[i, j]:.2f}', ha='center', va='center',
                color='white' if cm_norm[i, j] > 0.5 else 'black', fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'confusion_ensemble.pdf'), bbox_inches='tight')
plt.close(fig)
print('wrote confusion_ensemble.pdf')


# ---------- STATISTICS for the report ----------
print('\n=== val MAE / score per method ===')


def score(y, p):
    mae = mean_absolute_error(y, p)
    return 1 - mae / 4, mae


for name, p in [('xlmr_base', xlmr_val), ('mdeberta', mdeb_val), ('ensemble', ens_val)]:
    s_arg, m_arg = score(y_val, np.argmax(p, axis=1))
    s_med, m_med = score(y_val, median_round(p))
    print(f'  {name:10s} argmax: score={s_arg:.4f} mae={m_arg:.4f} | '
          f'median: score={s_med:.4f} mae={m_med:.4f}')

print('\n=== bootstrap 95% CI on ensemble (median) val score ===')
rng = np.random.default_rng(0)
n = len(y_val)
boot = []
for _ in range(1000):
    idx = rng.integers(0, n, n)
    pred = median_round(ens_val[idx])
    mae = mean_absolute_error(y_val[idx], pred)
    boot.append(1 - mae / 4)
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
print(f'  point: {1 - mean_absolute_error(y_val, pred_ens) / 4:.4f}  '
      f'95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]')

print('\n=== per-language MAE (heuristic DE/EN) ===')
_DE_CHARS = ('ä', 'ö', 'ü', 'ß')
_DE_WORDS = (' der ', ' die ', ' das ', ' und ', ' ich ', ' nicht ',
             ' ist ', ' ein ', ' eine ', ' mit ', ' für ')


def guess_lang(t):
    t = (t or '').lower()
    if any(c in t for c in _DE_CHARS):
        return 'de'
    tp = f' {t} '
    if any(w in tp for w in _DE_WORDS):
        return 'de'
    return 'en'


val_df = val_df.copy()
val_df['lang'] = val_df['sentence'].apply(guess_lang)
val_df['pred'] = pred_ens
for lang in ['de', 'en']:
    sub = val_df[val_df['lang'] == lang]
    mae = mean_absolute_error(sub['label'], sub['pred'])
    print(f'  {lang}: n={len(sub):5d}  MAE={mae:.4f}  score={1 - mae / 4:.4f}')

print('\n=== per-class accuracy and mean abs error ===')
for c in range(5):
    sub = val_df[val_df['label'] == c]
    acc = (sub['pred'] == c).mean()
    mae = mean_absolute_error(sub['label'], sub['pred'])
    print(f'  class {c} ({c + 1}★): n={len(sub):5d}  acc={acc:.3f}  mae={mae:.3f}')

print('\n=== fraction of extreme confusions (|err| >= 3) ===')
err = np.abs(y_val - pred_ens)
print(f'  |err|=0: {(err == 0).mean():.3f}')
print(f'  |err|=1: {(err == 1).mean():.3f}')
print(f'  |err|=2: {(err == 2).mean():.3f}')
print(f'  |err|=3: {(err == 3).mean():.3f}')
print(f'  |err|=4: {(err == 4).mean():.3f}')
