import h5py
import numpy as np
import pandas as pd


def load_data(path, subset):
    with h5py.File(path, 'r') as f:
        X = f[f'{subset}_traces/traces'][:]
        z = f[f'{subset}_traces/labels'][:]
        metadata = f[f'{subset}_traces/metadata'][:]

    return (
        pd.DataFrame(X.astype(np.float32), columns=range(X.shape[1])),
        pd.Series(z.astype(np.uint8), name='z'),
        pd.DataFrame({'plaintext': metadata['plaintext'][:, 2].astype(np.uint8),
                      'key': metadata['key'][:, 2].astype(np.uint8)})
    )
