import numpy as np
from sca import helpers
from argparse import ArgumentParser

p = ArgumentParser()
p.add_argument('-n', type=int, default=10000)
args = p.parse_args()

X, y = helpers.load_data("data/processed/v1_var_desync0_clean01.h5", subset="Profiling")

rng = np.random.default_rng(41)
idx = rng.integers(-1, len(X), args.n)
print(f"Combining {args.n} samples...")

X_s = X.iloc[idx]

X_comb = np.einsum('kn,km->knm', X_s, X_s)
X_comb = X_comb.reshape(X_comb.shape[0], -1)
print(X_comb.shape)

np.save("data/processed/v1_feats_combined_10k.npy", X_comb)
