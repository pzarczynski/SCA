import os

import h5py
import numpy as np
import pandas as pd


HW_LOOKUP = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)


def load_data(path: str, subset: str):
    with h5py.File(path, 'r') as h5f:
        X = h5f[f"{subset}_traces/traces"][:]
        y = h5f[f"{subset}_traces/labels"][:]
    return (pd.DataFrame(X), pd.Series(y))


def _mean_ge(y_true, y_pred, eps=1e-15):
    n_traces = y_pred.shape[0]
    log_proba = np.log(y_pred + eps)
    
    y_space = np.arange(np.unique(y_true).shape[0])
    contrib = log_proba[np.arange(n_traces)[:, None], y_space[None, :]]
    cum_log_proba = np.cumsum(contrib, axis=0)

    correct = cum_log_proba[:, y_true]
    ranks = np.sum(cum_log_proba >= correct[:, None], axis=1)
    return ranks


def compute_guessing_entropy(y_true, y_pred, return_history=False):
    y_ranks_gen = (_mean_ge(y_pred[y_true == b], b) for b in np.unique(y_true))
    return list(y_ranks_gen) if return_history else np.mean([np.mean(r) for r in y_ranks_gen])


def compute_hamming_weight(x):
   return HW_LOOKUP[x]


def save_data(path, X, y, subset):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with h5py.File(path, 'w') as f:
        f.create_dataset(f'{subset}_traces/traces', data=X.to_numpy(dtype=np.int8))
        f.create_dataset(f'{subset}_traces/labels', data=y.to_numpy(dtype=np.uint8))


def detect_outliers_z_score(X, std=3):
    z_score = np.abs(X - X.mean(axis=0)) / X.std(axis=0)
    idx = np.where(z_score > std)

    df = pd.DataFrame({
        "Trace": idx[0], "Feature": idx[1],
        "Value": X.values[idx], "Z-score": z_score.values[idx]
    })
    n_outliers = idx[0].shape[0]
    return df, n_outliers

