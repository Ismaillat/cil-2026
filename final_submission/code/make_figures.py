"""Generate every figure used in the report from preds/ and data/.

Run from the code/ directory:
    python make_figures.py
Reads data/train.csv, data/val_indices.npy, preds/*.npy and writes
the .pdf figures to figs/.
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, mean_absolute_error

ROOT = '.'
FIGDIR = 'figs'
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 200,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


# ----------------------------- Data + helpers -----------------------------

def median_round(p):
    return np.argmax(np.cumsum(p, axis=1) >= 0.5, axis=1)


_DE_CHARS = ('ä', 'ö', 'ü', 'ß')
_DE_WORDS = (' der ', ' die ', ' das ', ' und ', ' ich ', ' nicht ', ' ist ',
             ' ein ', ' eine ', ' mit ', ' für ', ' auch ', ' war ',
             ' es ', ' zu ', ' den ')


def _is_de(text):
    t = (text or '').lower()
    if any(c in t for c in _DE_CHARS):
        return True
    padded = f' {t} '
    return any(w in padded for w in _DE_WORDS)


train = pd.read_csv(os.path.join(ROOT, 'data', 'train.csv'))
val_idx = np.load(os.path.join(ROOT, 'data', 'val_indices.npy'))
val_df = train.loc[val_idx].reset_index(drop=True)
y_val = val_df['label'].values
val_lang = np.array(['de' if _is_de(s) else 'en' for s in val_df['sentence']])

preds_val = {n: np.load(os.path.join(ROOT, 'preds', f'{n}_val.npy'))
             for n in ['b1', 'b2', 'b5_gbm', 'b6_surface', 'bilstm', 'cnn_big',
                       'deberta_v3_large_en', 'gbert_large_de', 'mdeberta',
                       'xlmr_base']}

MODEL_LABELS = {
    'b1': 'TF--IDF',
    'b2': 'MiniLM',
    'b5_gbm': 'GBM',
    'b6_surface': 'Surface',
    'bilstm': 'BiLSTM',
    'cnn_big': 'CNN',
    'mdeberta': 'mDeBERTa',
    'xlmr_base': 'XLM-R',
    'deberta_v3_large_en': 'DeBERTa-EN',
    'gbert_large_de': 'gBERT-DE',
}
ORDER = ['b1', 'b2', 'b5_gbm', 'b6_surface', 'bilstm', 'cnn_big',
         'xlmr_base', 'mdeberta', 'deberta_v3_large_en', 'gbert_large_de']


# ----------------------------- 1. Surface features -----------------------------

n_excl = train['sentence'].str.count('!')
n_qst = train['sentence'].str.count(r'\?')
m_excl = n_excl.groupby(train['label']).mean()
m_qst = n_qst.groupby(train['label']).mean()

fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.6))
labels5 = ['1', '2', '3', '4', '5']
axes[0].bar(labels5, m_excl.values, color='#3b6fb5')
axes[0].set_title("Mean count of `!'")
axes[0].set_xlabel('star rating')
axes[0].set_ylim(0, max(m_excl.max(), 0.8) * 1.1)
axes[1].bar(labels5, m_qst.values, color='#3b6fb5')
axes[1].set_title("Mean count of `?'")
axes[1].set_xlabel('star rating')
axes[1].set_ylim(0, max(m_qst.max(), 0.1) * 1.15)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'surface_features.pdf'), bbox_inches='tight')
plt.close(fig)
print('wrote surface_features.pdf')


# ----------------------------- 2. Length distribution -----------------------------

nw = train['sentence'].str.split().str.len()
fig, ax = plt.subplots(figsize=(3.3, 1.7))
ax.hist(nw.clip(upper=300), bins=50, color='#3b6fb5',
        edgecolor='white', linewidth=0.3)
p99 = int(np.percentile(nw, 99))
ax.axvline(p99, color='#c4302b', linestyle='--', linewidth=1.0,
           label=f'99th pct = {p99}')
ax.axvline(256, color='black', linestyle=':', linewidth=1.0,
           label='max\\_length = 256')
ax.set_xlabel('words per review (clipped at 300)')
ax.set_ylabel('reviews')
ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'length_hist.pdf'), bbox_inches='tight')
plt.close(fig)
print('wrote length_hist.pdf')


# ----------------------------- 3. Per-language weights -----------------------------
# Weights produced by `python meta.py` on the canonical val split with the
# 10 base models active (full-fit on val_en / val_de, simplex search).
W_EN = {
    'b1': 0.028, 'b2': 0.044, 'b5_gbm': 0.050, 'b6_surface': 0.039,
    'bilstm': 0.000, 'cnn_big': 0.113, 'xlmr_base': 0.052,
    'mdeberta': 0.094, 'deberta_v3_large_en': 0.553, 'gbert_large_de': 0.026,
}
W_DE = {
    'b1': 0.044, 'b2': 0.000, 'b5_gbm': 0.007, 'b6_surface': 0.029,
    'bilstm': 0.000, 'cnn_big': 0.000, 'xlmr_base': 0.029,
    'mdeberta': 0.212, 'deberta_v3_large_en': 0.231, 'gbert_large_de': 0.448,
}

xs = np.arange(len(ORDER))
labels = [MODEL_LABELS[m] for m in ORDER]
w_en = [W_EN[m] for m in ORDER]
w_de = [W_DE[m] for m in ORDER]

fig, ax = plt.subplots(figsize=(3.3, 2.0))
bar_w = 0.4
ax.bar(xs - bar_w / 2, w_en, bar_w, label='English', color='#3b6fb5')
ax.bar(xs + bar_w / 2, w_de, bar_w, label='German', color='#c4302b')
ax.set_xticks(xs)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('learned weight')
ax.legend(loc='upper left', frameon=False)
ax.set_ylim(0, 0.62)
ax.grid(axis='y', alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'perlang_weights.pdf'), bbox_inches='tight')
plt.close(fig)
print('wrote perlang_weights.pdf')


# ----------------------------- 4. Per-language confusion matrices ----------------

# Reconstruct the per-lang test predictions on val using the OOF-equivalent
# (full-fit) weights, then per-language confusion.
def apply_weights(preds, w):
    return sum(w[i] * preds[i] for i in range(len(preds)))

w_en_arr = np.array([W_EN[m] for m in ORDER])
w_de_arr = np.array([W_DE[m] for m in ORDER])
val_arrs = [preds_val[m] for m in ORDER]

en_mask = val_lang == 'en'
de_mask = val_lang == 'de'

mix_en = apply_weights([p[en_mask] for p in val_arrs], w_en_arr)
mix_de = apply_weights([p[de_mask] for p in val_arrs], w_de_arr)
pred_en = median_round(mix_en)
pred_de = median_round(mix_de)

cm_en = confusion_matrix(y_val[en_mask], pred_en, labels=[0, 1, 2, 3, 4])
cm_de = confusion_matrix(y_val[de_mask], pred_de, labels=[0, 1, 2, 3, 4])
cm_en_n = cm_en / cm_en.sum(axis=1, keepdims=True)
cm_de_n = cm_de / cm_de.sum(axis=1, keepdims=True)

cmap = LinearSegmentedColormap.from_list(
    'softblue', [(1, 1, 1), (0.231, 0.435, 0.710)])

fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.75))
for ax, cm, title in zip(axes, [cm_en_n, cm_de_n],
                         ['English (n=12{,}871)', 'German (n=12{,}329)']):
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(['1', '2', '3', '4', '5'])
    ax.set_yticklabels(['1', '2', '3', '4', '5'])
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(title)
    for i in range(5):
        for j in range(5):
            v = cm[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=6, color='black' if v < 0.5 else 'white')
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'confusion_per_lang.pdf'), bbox_inches='tight')
plt.close(fig)
print('wrote confusion_per_lang.pdf')


# ----------------------------- 5. Error-correlation heat-map ----------------

errs = {m: median_round(preds_val[m]) != y_val for m in ORDER}
n = len(ORDER)
J = np.eye(n)
for i in range(n):
    for j in range(i + 1, n):
        a = errs[ORDER[i]]; b = errs[ORDER[j]]
        union = (a | b).sum()
        J[i, j] = J[j, i] = (a & b).sum() / max(1, union)

fig, ax = plt.subplots(figsize=(3.3, 2.5))
im = ax.imshow(J, cmap=cmap, vmin=0.15, vmax=0.85)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
labels_full = [MODEL_LABELS[m] for m in ORDER]
ax.set_xticklabels(labels_full, rotation=45, ha='right')
ax.set_yticklabels(labels_full)
for i in range(n):
    for j in range(n):
        v = J[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                fontsize=5.3, color='black' if v < 0.55 else 'white')
cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
cbar.set_label('Jaccard error overlap')
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'error_correlation.pdf'), bbox_inches='tight')
plt.close(fig)
print('wrote error_correlation.pdf')


# ----------------------------- 6. Per-language single-model bar chart ----

per_lang = {}
for m in ORDER:
    p = preds_val[m]
    pred = median_round(p)
    sc_en = 1 - mean_absolute_error(y_val[en_mask], pred[en_mask]) / 4
    sc_de = 1 - mean_absolute_error(y_val[de_mask], pred[de_mask]) / 4
    per_lang[m] = (sc_en, sc_de)

# (per_lang dict is built above; we print the scores at the end -- they
# back the EN/DE columns of Table 1 -- but no PDF is generated for
# them since the table already conveys the information.)


# ----------------------------- 7. Method progression -----------------------------

# All numbers are validation scores measured on the canonical 10% hold-out.
prog = [
    ('TF--IDF + LogReg', 0.882),
    ('mDeBERTa-v3-base', 0.907),
    ('mDeBERTa + XLM-R (uniform)', 0.908),
    ('Hybrid (specialists only)', 0.911),
    ('Weighted-MAE per-language', 0.913),
]
labs = [p[0] for p in prog]
vals = [p[1] for p in prog]

fig, ax = plt.subplots(figsize=(3.3, 2.2))
ys = np.arange(len(prog))
ax.barh(ys, vals, color=['#bababa', '#bababa', '#7fa6d4', '#3b6fb5', '#1e4570'])
ax.set_yticks(ys)
ax.set_yticklabels(labs)
ax.set_xlim(0.86, 0.925)
ax.axvline(0.906, color='#c4302b', linestyle='--', linewidth=1.0,
           label='grade-6 baseline = 0.906')
for y, v in zip(ys, vals):
    ax.text(v + 0.0007, y, f'{v:.3f}', va='center', fontsize=7)
ax.set_xlabel('validation score $1 - \\mathrm{MAE}/4$')
ax.legend(loc='lower right', frameon=False, fontsize=6.5,
          bbox_to_anchor=(1.0, 1.01))
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, 'progression.pdf'), bbox_inches='tight')
plt.close(fig)
print('wrote progression.pdf')


# ----------------------------- Print quick stats -----------------------------

print('\n--- per-language single-model scores (val) ---')
for m in ORDER:
    en, de = per_lang[m]
    print(f'  {MODEL_LABELS[m]:14s}  EN={en:.4f}  DE={de:.4f}')

print('\n--- error correlation matrix (Jaccard) ---')
for i, mi in enumerate(ORDER):
    print(f'  {MODEL_LABELS[mi]:14s} ' +
          ' '.join(f'{J[i, j]:.3f}' for j in range(n)))
