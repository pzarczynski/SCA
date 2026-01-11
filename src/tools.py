import os

import h5py
import numpy as np
import pandas as pd


def load_data(path, test=False, return_metadata=False):
    subset = "Attack" if test else "Profiling"

    with h5py.File(path, 'r') as f:
        X = f[f'{subset}_traces/traces'][:]
        labels = f[f'{subset}_traces/labels'][:]

        ret = [
            pd.DataFrame(X.astype(np.int8), columns=range(X.shape[1])),
            pd.Series(labels.astype(np.uint8), name='label')
        ]

        if return_metadata:
            ret.append(pd.DataFrame(metadata))
            metadata = f[f'{subset}_traces/metadata'][:]

    return tuple(ret)


def init_aes_sbox():
    sbox = np.zeros(256, dtype=np.uint8)
    p = q = 1

    while True:
        p = (p ^ (p << 1) ^ (0x1B if (p & 0x80) else 0)) & 0xFF
        q = (q ^ (q << 1) ^ (q << 2) ^ (q << 4) ^ (0x09 if (q & 0x80) else 0)) & 0xFF

        xformed = q ^ np.roll(np.array([q]), 1)[0] ^ np.roll(np.array([q]), 2)[0]
        xformed ^= np.roll(np.array([q]), 3)[0] ^ np.roll(np.array([q]), 4)[0]
        sbox[p] = (xformed ^ 0x63) & 0xFF

        if p == 1: break

    sbox[0] = 0x63
    return sbox

# SBOX_TABLE = init_aes_sbox()


def _mean_guessing_entropy(proba, y):
    N = proba.shape[0]
    log_proba = np.log(proba + 1e-30)

    contrib = log_proba[np.arange(N)[:, None], np.arange(256)[None, :]]
    cum_log_proba = np.cumsum(contrib, axis=0)

    correct = cum_log_proba[:, y]
    ranks = np.sum(cum_log_proba >= correct[:, None], axis=1)
    return ranks


def compute_score(p, y, return_history=False):
    ranks = [_mean_guessing_entropy(p[y == b], y[y == b]) for b in np.unique(y)]
    if return_history:
        return ranks
    return np.mean([np.mean(r) for r in ranks])


HW_LOOKUP = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)


def hamming_weight(x):
    return HW_LOOKUP[x]


def save_data(path, X, y, subset='Profiling'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset(f'{subset}_traces/traces', data=X.to_numpy(dtype=np.int8))
        f.create_dataset(f'{subset}_traces/labels', data=y.to_numpy(dtype=np.uint8))


def detect_outliers(X, std=3):
    z_score = np.abs(X - X.mean(0)) / X.std(0)
    idx = np.where(z_score > std)

    n_outliers = idx[0].shape[0]
    df = pd.DataFrame({"Trace": idx[0], "Feature": idx[1],
                       "Value": X.values[idx], "Z-score": z_score.values[idx]})
    return df, n_outliers
