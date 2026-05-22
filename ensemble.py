import glob
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_absolute_error


def median_round(probs):
    return np.argmax(np.cumsum(probs, axis=1) >= 0.5, axis=1)


def main():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    val_idx = np.load('data/val_indices.npy')
    y_val = train.loc[val_idx, 'label'].values

    val_files = sorted(glob.glob('preds/*_val.npy'))
    if not val_files:
        raise SystemExit('no preds/*_val.npy found; train models first')

    names = [os.path.basename(f).replace('_val.npy', '') for f in val_files]
    print('models:', names)

    val_probs = [np.load(f) for f in val_files]
    for name, p in zip(names, val_probs):
        mae = mean_absolute_error(y_val, median_round(p))
        print(f'  {name:24s} val score={1 - mae / 4:.4f}')

    ens_val = np.mean(val_probs, axis=0)
    mae = mean_absolute_error(y_val, median_round(ens_val))
    print(f'  {"ENSEMBLE":24s} val score={1 - mae / 4:.4f}  mae={mae:.4f}')
    print(confusion_matrix(y_val, median_round(ens_val), labels=[0, 1, 2, 3, 4]))

    test_files = [f.replace('_val.npy', '_test.npy') for f in val_files]
    ens_test = np.mean([np.load(f) for f in test_files], axis=0)
    pred = median_round(ens_test)

    os.makedirs('submissions', exist_ok=True)
    pd.DataFrame({'id': test['id'], 'label': pred.astype(int)}).to_csv(
        'submissions/ensemble.csv', index=False)
    print('wrote submissions/ensemble.csv')


if __name__ == '__main__':
    main()
