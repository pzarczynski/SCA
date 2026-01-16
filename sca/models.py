import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline

from sca.tools import compute_score


def anovaf_classifier(
    estimator, n_splits: int, seed: int,
    param_grid: dict, verbose: int = 0
):
    pipeline = make_pipeline(SelectKBest(f_classif), estimator)

    scorer = make_scorer(
        compute_score, greater_is_better=False,
        response_method="predict_proba"
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=dict(param_grid),
        scoring=scorer,
        cv=kf,
        verbose=verbose,
        return_train_score=True,
        n_jobs=-1
    )

    def fn(X, y):
        gs.fit(X, y)
        return pd.DataFrame(gs.cv_results_)

    return fn
