import numpy as np
from numba import njit, prange
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted


@njit(parallel=True, fastmath=True)
def _inter_f_stat_kernel(X, y, n_classes):
    n, m = X.shape
    num_inter = (m * (m - 1)) // 2
    total_features = 3 * num_inter

    sum_c = np.zeros((n_classes, total_features))
    sq_sum_c = np.zeros((n_classes, total_features))
    n_c = np.zeros(n_classes)

    for i in range(n):
        c = y[i]
        n_c[c] += 1

        idx = 0
        for j in range(m):
            v_j = X[i, j]
            for k in range(j + 1, m):
                v_k = X[i, k]

                f1, f2, f3 = v_j * v_k, v_j + v_k, (v_j - v_k)**2

                sum_c[c, idx] += f1
                sq_sum_c[c, idx] += f1**2

                sum_c[c, num_inter + idx] += f2
                sq_sum_c[c, num_inter + idx] += f2**2

                sum_c[c, 2 * num_inter + idx] += f3
                sq_sum_c[c, 2 * num_inter + idx] += f3**2

                idx += 1

    return sum_c, sq_sum_c, n_c


def _get_top_k_indices(sum_c, sq_sum_c, n_c, k):
    n_total = np.sum(n_c)
    mean_total = np.sum(sum_c, axis=0) / n_total
    mean_c = sum_c / n_c[:, None]

    ss_between = np.sum(n_c[:, None] * (mean_c - mean_total)**2, axis=0)
    ss_within = np.sum(sq_sum_c - (sum_c**2 / n_c[:, None]), axis=0)

    df_between = len(n_c) - 1
    df_within = n_total - len(n_c)

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    f_values = ms_between / ms_within
    f_values = np.nan_to_num(f_values)

    return np.argsort(f_values)[-k:]


@njit(parallel=True, fastmath=True)
def _transform_kernel(X, out, selected_lookup, num_inter):
    n_samples, m = X.shape
    k_features = out.shape[1]

    pair_to_jk = np.empty((num_inter, 2), dtype=np.int32)
    curr = 0
    for j in range(m):
        for k in range(j + 1, m):
            pair_to_jk[curr, 0] = j
            pair_to_jk[curr, 1] = k
            curr += 1

    for i in prange(n_samples):
        for f in range(k_features):
            op_type = selected_lookup[f, 0]
            pair_idx = selected_lookup[f, 1]

            j = pair_to_jk[pair_idx, 0]
            k = pair_to_jk[pair_idx, 1]

            v_j = X[i, j]
            v_k = X[i, k]

            if op_type == 0:
                out[i, f] = v_j * v_k
            elif op_type == 1:
                out[i, f] = (v_j + v_k)**2
            else:
                out[i, f] = (v_j - v_k)**2

class InteractionsSelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.int64)

        n, m = X.shape
        self.num_inter_ = (m * (m - 1)) // 2
        classes = np.unique(y)
        n_classes = len(classes)

        sum_c, sq_sum_c, n_c = _inter_f_stat_kernel(X, y, n_classes)

        self.selected_indices_ = _get_top_k_indices(sum_c, sq_sum_c, n_c, self.k)

        self.selected_lookup_ = np.empty((self.k, 2), dtype=np.int32)
        for i, idx in enumerate(self.selected_indices_):
            self.selected_lookup_[i, 0] = idx // self.num_inter_ # 0:mul, 1:add, 2:sq_diff
            self.selected_lookup_[i, 1] = idx % self.num_inter_

        return self

    def transform(self, X):
        check_is_fitted(self)
        X = np.ascontiguousarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        out = np.empty((n_samples, self.k), dtype=np.float64)
        _transform_kernel(X, out, self.selected_lookup_, self.num_inter_)
        return out
