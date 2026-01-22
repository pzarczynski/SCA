import os

import h5py
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

SBOX = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
], dtype=np.uint8)


def pi_score(log_proba, pts, k):
    z = SBOX[pts ^ k]
    correct_proba = log_proba[np.arange(len(pts)), z] 
    return 8 + np.mean(correct_proba)


def PI(proba, pts, ks, eps=1e-40):
    pts, ks = pts[:, 2], ks[:, 2]
    log_proba = np.log2(np.maximum(proba, eps))
    return np.array([pi_score(log_proba[ks == k], pts[ks == k], k) 
                     for k in range(256)])


def load_data(path: str, as_df=False, attack=False):
    subset = "Attack" if attack else "Profiling"
    
    with h5py.File(path, 'r') as f:
        X = f[f"{subset}_traces/traces"][:].astype(np.float32)
        y = f[f"{subset}_traces/labels"][:].astype(np.uint8)
        meta = f[f"{subset}_traces/metadata"][:]
        pts = meta["plaintext"].astype(np.uint8)
        ks = meta["key"].astype(np.uint8)

    if as_df:
        return (pd.DataFrame(X).rename(columns=str),  pd.Series(y), 
                pd.DataFrame(pts), pd.DataFrame(ks))
        
    return X, y, pts, ks


def save_data(path, X, y, pts, ks, subset="Profiling"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    meta = np.empty(X.shape[0], dtype=np.dtype([
        ("plaintext", np.uint8, (16,)), ("key", np.uint8, (16,)),
    ]))
    meta["plaintext"], meta["key"] = pts, ks

    with h5py.File(path, 'w') as f:
        grp = f.create_group(f"{subset}_traces")
        grp.create_dataset('traces', data=np.asarray(X, dtype=np.int8))
        grp.create_dataset('labels', data=np.asarray(y, dtype=np.uint8))
        grp.create_dataset('metadata', data=meta)


def cv(
    model, X, y, pts, ks, n_repeats=5, n_splits=2, 
    verbose=True, seed=None, skip_last=False
):       
    X = pd.DataFrame(X)
    y = pd.Series(y)
    pts = pd.DataFrame(pts)
    ks = pd.DataFrame(ks)

    kf = RepeatedStratifiedKFold(
            n_splits=n_splits, 
            n_repeats=n_repeats, 
            random_state=seed)
    cumscore = 0
    
    for i, (prof_idx, atk_idx) in enumerate(kf.split(X, y)):
        if skip_last and (i + 1) % n_splits == 0: continue

        X_prof, y_prof = X.iloc[prof_idx], y.iloc[prof_idx]
        X_atk, pts_atk, ks_atk = (X.iloc[atk_idx], pts.iloc[atk_idx], 
                                  ks.iloc[atk_idx])
    
        model.fit(X_prof, y_prof)
        probs = model.predict_proba(X_atk)
        
        scores = PI(probs, pts_atk, ks_atk)
        cumscore += np.mean(scores)

        if verbose: logging.info(f"[{i + 1}] Mean PI: {np.mean(scores):.3f}")
        
    return cumscore / (n_splits * n_repeats) 


def feature_importances(model, X, y, pts, ks, seed=None):
    _ = cv(model, X, y, pts, ks, n_repeats=1, skip_last=True, seed=seed)    
    rf = model.named_steps['randomforestclassifier']
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]
    return importances, idx


def n_feats_search(model, X, y, pts, ks, feat_rank, n_list, n_repeats=5, seed=None):
    results = {'n_feats': [], 'PI': []}
    prev_score = None

    for n_feats in n_list:
        score = cv(
            model, X[:, feat_rank[:n_feats]], y, pts, ks,
            n_repeats=n_repeats, verbose=False, seed=seed,
        )

        logging.info(f"Mean PI (n_feats={n_feats}): {score:.2e}")

        results['n_feats'].append(n_feats)
        results['PI'].append(score)

        if prev_score is not None and prev_score > score:
            break

        prev_score = score

    return pd.DataFrame(results)



def split_idx(X, y, seed=None):
    return next(StratifiedKFold(2, shuffle=True, random_state=seed).split(X, y))


def rf_feature_importances(X, y, pts, ks, seed, max_depth=5):
    pl = make_pipeline(
        StandardScaler(),
        RFC(max_depth=max_depth, n_jobs=-1, min_samples_leaf=10, random_state=seed)
    )
    return feature_importances(pl, X, y, pts, ks, seed=seed)


@dataclass(eq=False)
class LGBValidateEvery:
    X_atk: np.ndarray
    pts_atk: np.ndarray
    ks_atk: np.ndarray
    patience: int = field(default=50)
    check_every: int = field(default=10)

    def __post_init__(self):
        self.best_score = -float('inf')
        self.best_iter = 0
        self.no_improvement_count = 0
        self.history = []

    def __call__(self, env):
        iteration = env.iteration

        if (iteration + 1) % self.check_every != 0:
            return

        proba = env.model.predict(self.X_atk, num_iteration=iteration)
        proba = proba.reshape(len(self.X_atk), 256, order='F')

        score = np.mean(PI(proba, self.pts_atk, self.ks_atk))
        self.history.append((iteration, score))

        if score > self.best_score:
            self.best_score = score
            self.best_iter = iteration
            self.no_improvement_count = 0
            print(f"[SCA] Iter {iteration+1}: NEW BEST Score: {score:.4f}")
        else:
            self.no_improvement_count += 1
            print(f"[SCA] Iter {iteration+1}: Score: {score:.4f} (No improv: {self.no_improvement_count})")

        if self.no_improvement_count >= (self.patience / self.check_every):
            print(f"Early stopping at iteration {iteration+1}. Best score: {self.best_score:.4f} at {self.best_iter+1}")
            raise lgb.callback.EarlyStopException(self.best_iter, self.best_score)
