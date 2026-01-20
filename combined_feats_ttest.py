import numpy as np
from scipy import stats
from sca import helpers as h
from numba import njit, prange
from numba_progress import ProgressBar


@njit
def welch_ttest(x, y):
    n1, n2 = len(x), len(y)
    mean1, mean2 = x.mean(), y.mean()
    var1, var2 = x.var(), y.var()
    
    se = np.sqrt(var1 / n1 + var2 / n2)
    assert se > 0
    t_stat = (mean1 - mean2) / se
    return t_stat

@njit(nogil=True, parallel=True)
def compute_comb_tstats(X, mask, prog_proxy):
    n_features = X.shape[1]
    n_pairs = n_features * (n_features - 1) // 2
    results = np.empty(n_pairs, dtype=np.float64)
    
    for idx in prange(n_pairs):
        i = 0
        pairs_before = 0
        while pairs_before + (n_features - i - 1) <= idx:
            pairs_before += n_features - i - 1
            i += 1
        j = idx - pairs_before + i + 1
        
        X_comb = X[:, i] * X[:, j]
        r = welch_ttest(X_comb[mask], X_comb[~mask])
        
        if r > 4.5:
            print("!!! >4.5 (", i, ",", j, "):", r, "!!!")

        results[idx] = r
        prog_proxy.update(1)
    
    return results


if __name__ == '__main__':
    X, y = h.load_data('data/processed/v1_var_desync0_clean.h5', 
                             subset="Profiling")
    X = np.asarray(X, dtype=np.float32)
    X -= np.mean(X, axis=0, keepdims=True)
    mask = h.hamming_weight(y) >= 4

    m = X.shape[1]
    
    with ProgressBar(total=m * (m-1) // 2) as p:
        out = compute_comb_tstats(X, mask, p)

    np.save("data/processed/comb_feats_ttest.npy", out)
