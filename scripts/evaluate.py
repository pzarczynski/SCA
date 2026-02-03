import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from scripts import plots, util
from scripts.cv import eval_model

SEED = 42


pipeline = make_pipeline(
    SelectFromModel(
        RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=500,
            random_state=SEED,
            n_jobs=-1,
        ),
        max_features=100,
    ),
    StandardScaler(),
    PolynomialFeatures(degree=(2, 2), interaction_only=True, include_bias=False),
    SelectKBest(score_func=f_classif, k=750),
    LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', n_components=5),
    LogisticRegression(C=1e-2, random_state=SEED)
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
