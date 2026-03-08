import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin, BaseEstimator, check_is_fitted
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import QuantileTransformer

from functools import partial
from scripts import plots, util
from scripts.cv import eval_model

from sca_fused_ops import fused_select_poly, fused_transform_poly

SEED = 42


class FusedSelectPolynomial(BaseEstimator, TransformerMixin):
    def __init__(self, select: int, k: int = 0) -> None:
        self.select = select
        self.k = k

    def fit(self, X, y):
        idx, self.means, self.scales = fused_select_poly(
            X.astype(np.float32), 
            y.astype(np.uint64),
            int(np.max(y)) + 1, 
            int(self.select), 
            int(self.k),
        )
        _, self.ix, self.jx = map(partial(np.array, dtype=np.uint64), zip(*idx))
        return self

    def transform(self, X):
        check_is_fitted(self, ("ix", "jx", "means", "scales"))
        out = fused_transform_poly(
            np.asarray(X, dtype=np.float32), 
            self.means, 
            self.scales, 
            self.ix,
            self.jx,
        )
        return out

pipeline = make_pipeline(
    SelectFromModel(
        ExtraTreesClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=500,
            max_features='sqrt',
            random_state=SEED,
            n_jobs=-1,
        ),
        max_features=100,
    ),
    FusedSelectPolynomial(select=750),
    # PowerTransformer(method='yeo-johnson'),
    LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', n_components=5),
    LogisticRegression(C=1e-2, random_state=SEED)
    # GaussianNB()
)


if __name__ == "__main__":
    X, y, pts, ks, _ = util.load_data('data/processed/ASCADv.h5')
    X_atk, _, pts_atk, ks_atk, _ = util.load_data('data/raw/ASCADv.h5', attack=True)

    results, aux = eval_model(
        SEED, pipeline,
        X, y, pts, ks,
        X_atk, pts_atk, ks_atk,
        score_fn=util.compute_pge,
        n_repeats=10
    )

    mr_atk = np.array([a[0] for a in aux[1]])
    frs = [int(a[1]) for a in aux[1]]

    fig, ax = plt.subplots(figsize=(8, 6))
    plots.plot_mean_std(mr_atk[:, :1000], ax=ax)
    ax.axvline(np.mean(frs), color='red', linestyle='--')
    ax.axhline(128, color='blue', linestyle='--')
    ax.axhline(1, color='green', linestyle='--')
    ax.set_ylim(-64, 256)
    plots.labs(ax, "Number of traces", "Partial Guessing Entropy", title="Poly + LDA + LR on ASCADv1")
    ax.legend(["Mean Partial Guessing Entropy",
                "Standard deviation",
                f"Mean traces needed ({np.mean(frs):.0f})",
                "Random guessing",
                "Key rank = 1"])
    plots.savetight(fig, "09_pl_ascadv1")
    plt.show()
