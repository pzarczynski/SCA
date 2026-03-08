from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

from scripts.evaluate import FusedSelectPolynomial

SEED = 42


pipeline_naive = make_pipeline(
    SelectFromModel(
        RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=500,
            random_state=SEED,
            n_jobs=-1,
        ),
        max_features=100,
    ),
    StandardScaler(),
    PolynomialFeatures(degree=(2, 2), interaction_only=True, include_bias=False),
    SelectKBest(score_func=f_classif, k=500),
    LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', n_components=5),
    LogisticRegression(C=1e-2, random_state=SEED)
)


pipeline_fused = make_pipeline(
    SelectFromModel(
        ExtraTreesClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=500,
            random_state=SEED,
            n_jobs=-1,
        ),
        max_features=100,
    ),
    FusedSelectPolynomial(750, k=0),
    LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', n_components=5),
    LogisticRegression(C=1e-2, random_state=SEED)
)


if __name__ == "__main__":
    import memray
    from time import perf_counter
    from scripts import util

    X, y, *_ = util.load_data()

    with memray.Tracker("naive_pipeline.bin"): pipeline_naive.fit(X, y)
    with memray.Tracker("fused_pipeline.bin"): pipeline_fused.fit(X, y)

    start = perf_counter()
    pipeline_fused.fit(X, y)
    print(f"Fused pipeline took {perf_counter() - start:.4f} seconds")

    start = perf_counter()
    pipeline_naive.fit(X, y)
    print(f"Naive pipeline took {perf_counter() - start:.4f} seconds")