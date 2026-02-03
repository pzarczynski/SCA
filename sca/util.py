import logging
import os
from functools import partial
from time import perf_counter

import fastcluster as fc
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.ops import segment_sum
from natsort import natsorted
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from sklearn.model_selection import RepeatedStratifiedKFold as RSKF
from tqdm.auto import tqdm

jax.config.update("jax_platform_name", "cpu")

RSKF2 = partial(RSKF, n_splits=2)


def init_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
        datefmt="%H:%M:%S",
    )

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

SBOX_JAX = jnp.array(SBOX)


@jax.jit
def compute_pi(log2_proba, pts, ks):
    n_traces, n_classes = log2_proba.shape
    correct = log2_proba[jnp.arange(n_traces), SBOX_JAX[pts ^ ks]]

    sum = segment_sum(correct, ks, n_classes)
    den = segment_sum(jnp.ones_like(correct), ks, n_classes)
    return 8.0 + (sum / jnp.maximum(den, 1.0))


def load_data(path='data/processed/ASCADv.h5', attack=False, tgt_only=True):
    subset = "Attack" if attack else "Profiling"

    with h5py.File(path, 'r') as f:
        X = np.ascontiguousarray(f[f"{subset}_traces/traces"][:].astype(np.float32))
        y = np.ascontiguousarray(f[f"{subset}_traces/labels"][:].astype(np.uint8))
        pts = np.ascontiguousarray(f[f"{subset}_traces/metadata"]["plaintext"].astype(np.uint8))
        ks = np.ascontiguousarray(f[f"{subset}_traces/metadata"]["key"].astype(np.uint8))
        masks = np.ascontiguousarray(f[f"{subset}_traces/metadata"]["masks"].astype(np.uint8))

    if tgt_only:
        pts, ks, masks = pts[:, 2], ks[:, 2], masks[:, 2]

    return X, y, pts, ks, masks


def save_data(path, X, y, pts, ks, masks, attack=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    subset = "Attack" if attack else "Profiling"

    meta_dtype = np.dtype([("plaintext", np.uint8, (pts.shape[1],)),
                           ("key",       np.uint8, (ks.shape[1],)),
                           ("masks",     np.uint8, (masks.shape[1],))])
    meta = np.empty(X.shape[0], dtype=meta_dtype)
    meta["plaintext"], meta["key"], meta["masks"] = pts, ks, masks

    with h5py.File(path, 'w') as f:
        grp = f.create_group(f"{subset}_traces")
        grp.create_dataset('traces', data=np.asarray(X, dtype=np.int8))
        grp.create_dataset('labels', data=np.asarray(y, dtype=np.uint8))
        grp.create_dataset('metadata', data=meta)


def sample_trace(X, y, byte_val, seed=42):
    mask = y == byte_val
    rng = np.random.default_rng(seed=seed)
    idx = rng.integers(0, np.sum(mask))
    trace = X[mask][idx]
    return trace


def hw(x):
    return np.bitwise_count(x)


def results_to_dict(df, **kwargs):
    return {**kwargs,
            'prof_score_mean':  df['prof_score_mean'].mean(),
            'prof_score_std':   df['prof_score_mean'].std(),
            'atk_score_mean':   df['atk_score_mean'].mean(),
            'atk_score_std':    df['atk_score_mean'].std()}


def _pge(log_proba2, pts, ks):
    n_traces, n_classes = log_proba2.shape
    z_hyp = SBOX[pts[:, None] ^ np.arange(n_classes)[None, :]]

    allproba = log_proba2[np.arange(n_traces)[:, None], z_hyp]
    cumproba = np.cumsum(allproba, axis=0)
    all_pge = np.argsort(-cumproba, axis=1)

    correct_ranks = np.where(all_pge == ks[:, None])[1]
    first_rank0 = np.where(correct_ranks == 0)[0]

    if first_rank0.size == 0:
        first_rank0 = n_traces - 1

    return correct_ranks + 1, first_rank0 + 1


def compute_pge(log_proba2, pts, ks):
    all_pge, fr0 = zip(*(_pge(log_proba2[ks == k], pts[ks == k], ks[ks == k])
                         for k in np.unique(ks)))
    minlen = np.min(np.array([r.shape[0] for r in all_pge]))
    all_pge = np.array([r[:minlen] for r in all_pge])
    return np.mean(all_pge, axis=0), np.mean(fr0)


def save_results(gs, name):
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(gs).to_csv(f'data/results/{name}.csv', index=False)


def mean_std(X):
    return X.mean(axis=0), X.std(axis=0)


def transform_split(t, split):
    split[0][0] = t.fit_transform(split[0][0], split[0][1])
    split[1][0] = t.transform(split[1][0])
    return split


def precompute_splits(seed, t, X, y, pts, ks, masks, n_repeats=5):
    splits = ([[X[p], y[p], pts[p], ks[p], masks[p]],
               [X[a], y[a], pts[a], ks[a], masks[a]]] for (p, a) in
              RSKF2(n_repeats=n_repeats, random_state=seed).split(X, y))
    return [transform_split(t, split) for split in tqdm(splits, total=2*n_repeats)]


def transform_splits(t, splits):
    return [transform_split(t, split) for split in tqdm(splits, total=len(splits))]


def inspect_pipeline(pl):
    if 'clusterfeatures' in pl.named_steps:
        clusterer = pl.named_steps['clusterfeatures']
        clus_repr = clusterer.repr_idx_ if hasattr(clusterer, 'repr_idx_') else None

    selected_idx = np.where(pl.named_steps['selectfrommodel'].get_support())[0]

    poly_out = pl.named_steps['polynomialfeatures'].get_feature_names_out()
    best_mask = pl.named_steps['selectkbest'].get_support()
    scores = pl.named_steps['selectkbest'].scores_

    parents = [y.split(' ') for y in poly_out[best_mask]]
    parents = [[int(a[1:]), int(b[1:])] for a, b in parents]
    pairs = [[selected_idx[a], selected_idx[b]] for (a, b) in parents]

    pairs = np.array(pairs)
    if 'clusterfeatures' in pl.named_steps and clus_repr is not None:
        pairs = pairs[clus_repr]
        scores = scores[clus_repr]
    feats = natsorted(set(pairs[:, 0]).union(set(pairs[:, 1])))
    return {
        'selected_features': feats,
        'scores': scores,
        'cluster_repr': clus_repr if 'clusterfeatures' in pl.named_steps else None,
        'selected_rf': selected_idx,
        'pairs': pairs,
    }


class ClusterFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.02, n_clusters=None, method='average', cluster_fn=None):
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.method = method
        self.cluster_fn = cluster_fn

    def _compute_representatives(self, C):
        clusters_dict = {}
        for i, cid in enumerate(self.clusters):
            clusters_dict.setdefault(cid, []).append(i)

        ordered = [clusters_dict[k] for k in sorted(clusters_dict.keys())]
        repr_idx = []
        for feat_idx in ordered:
            feat_idx = np.array(feat_idx)
            if len(feat_idx) == 1:
                repr_idx.append(feat_idx[0])
            else:
                sub = C[np.ix_(feat_idx, feat_idx)]
                scores = sub.sum(axis=1) - 1.0
                repr_idx.append(feat_idx[np.argmax(scores)])
        return np.array(repr_idx)

    def _get_cluster_indices(self):
        clusters_dict = {}
        for i, cid in enumerate(self.clusters):
            clusters_dict.setdefault(cid, []).append(i)
        return [clusters_dict[k] for k in sorted(clusters_dict.keys())]

    def fit(self, X, y=None):
        logging.info("Clustering features...")
        start_time = perf_counter()

        C = np.corrcoef(X, rowvar=False)
        n_feat = C.shape[0]

        i, j = np.triu_indices(n_feat, k=1)
        dist = np.abs(1.0 - C[i, j])

        Z = fc.linkage(dist, method=self.method)

        if self.n_clusters is not None:
            self.clusters = fcluster(Z, t=self.n_clusters, criterion='maxclust')
        else:
            self.clusters = fcluster(Z, t=self.threshold, criterion='distance')

        self.cluster_indices_ = self._get_cluster_indices()
        self.repr_idx_ = self._compute_representatives(C)
        self.n_clusters_ = len(np.unique(self.clusters))

        total_time = perf_counter() - start_time
        logging.info(f"Clustering complete (took {total_time:.2f} seconds), "
                     f"created {self.n_clusters_} clusters.")
        return self

    def transform(self, X):
        check_is_fitted(self, 'cluster_indices_')

        if self.cluster_fn is None:
            return X[:, self.repr_idx_]
        else:
            transformed = []
            for feat_indices in self.cluster_indices_:
                cluster_features = X[:, feat_indices]
                cluster_transformed = self.cluster_fn(cluster_features)
                if cluster_transformed.ndim == 1:
                    cluster_transformed = cluster_transformed[:, None]
                if cluster_transformed.shape[0] != X.shape[0]:
                    raise ValueError(
                        "cluster_fn must return array with same number of rows as input"
                    )
                transformed.append(cluster_transformed)
            return np.hstack(transformed)
