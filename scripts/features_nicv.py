import numpy as np
from numba import njit, prange
from numba_progress import ProgressBar


@njit(nogil=True, parallel=True)
def combined_nicv(X, y, pbar):
    n_feat = X.shape[1]
    result = np.empty((n_feat, n_feat))
    y_space = np.unique(y)
    n_classes = y_space.shape[0]

    for i in prange(n_feat):
        for j in range(i + 1):
            feat_ij = X[:, i] * X[:, j]

            means_sum = 0.0
            means_sq_sum = 0.0

            for k in range(n_classes): 
                subset = feat_ij[y == y_space[k]]
                m = np.mean(subset)
                means_sum += m
                means_sq_sum += m * m

            var_means = (means_sq_sum / n_classes) - (means_sum / n_classes)**2
            var_total = np.var(feat_ij)
            score = var_means / var_total

            result[j, i] = result[i, j] = score
            pbar.update(1)

    return result


if __name__ == '__main__':
    from sca import helpers
    X, y, *_ = helpers.load_data('data/processed/ascadv_clean.h5')
    # hw = np.bitwise_count(y)
    total = X.shape[1] * (X.shape[1] + 1) // 2

    with ProgressBar(total=total) as pbar:
        result = combined_nicv(X, y, pbar)
    
    np.save('data/combined_nicv_results.npy', result)
