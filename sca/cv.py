import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone

from sca import util


def _cv_single_fold(
    i, base_model, score_fn,
    X_prof, y_prof, pts_prof, ks_prof,
    X_atk, pts_atk, ks_atk, verbose
):
    util.init_logger()

    model = clone(base_model)
    model.fit(X_prof, y_prof)

    proba_prof = model.predict_proba(X_prof)
    log2_proba_prof = np.log2(proba_prof + 1e-15)
    scores_prof = score_fn(log2_proba_prof, pts_prof, ks_prof)

    proba_atk = model.predict_proba(X_atk)
    log2_proba = np.log2(proba_atk + 1e-15)
    scores_atk = score_fn(log2_proba, pts_atk, ks_atk)

    if verbose:
        logging.info(f"FOLD {i+1}: "
                     f"PROF={np.mean(scores_prof):.4f}; "
                     f"ATK={np.mean(scores_atk):.4f}")

    return (np.mean(scores_prof), np.std(scores_prof),
            np.mean(scores_atk),  np.std(scores_atk))


def _results_to_df(results):
    (scores_prof_mean, scores_prof_std,
     scores_atk_mean, scores_atk_std) = zip(*results)
    return pd.DataFrame({'prof_score_mean': np.array(scores_prof_mean),
                         'prof_score_std' : np.array(scores_prof_std),
                         'atk_score_mean' : np.array(scores_atk_mean),
                         'atk_score_std'  : np.array(scores_atk_std)})

def cv(
    seed, model,
    X, y, pts, ks, masks=None,
    n_repeats=5, preselect=200,
    verbose=True, n_jobs=1,
):
    kf = util.RSKF2(n_repeats=n_repeats, random_state=seed)
    jobs = []

    for i, (prof_idx, atk_idx) in enumerate(kf.split(X, ks)):
        X_prof, y_prof, pts_prof, ks_prof = X[prof_idx], y[prof_idx], pts[prof_idx], ks[prof_idx]
        X_atk, pts_atk, ks_atk = X[atk_idx], pts[atk_idx], ks[atk_idx]

        if masks is not None and preselect > 0:
            raw_hw = util.hw(util.SBOX[ks ^ pts] ^ masks)
            rho, _ = stats.pearsonr(X, raw_hw[:, None])
            idx = np.argsort(-rho)[:preselect]
            X_prof, X_atk = X_prof[:, idx], X_atk[:, idx]

        jobs.append(
            delayed(_cv_single_fold)(
                i, model, util.compute_pi,
                X_prof, y_prof, pts_prof, ks_prof,
                X_atk, pts_atk, ks_atk, verbose=verbose
            )
        )

    return _results_to_df(Parallel(
        n_jobs=n_jobs, backend='loky',
        verbose=1 if verbose else 0,
    )(jobs))


def cv_precomputed(
    model, splits,
    verbose=True, n_jobs=1,
):
    jobs = []

    for i, ((X_prof, y_prof, pts_prof, ks_prof, _),
            (X_atk, y_atk, pts_atk, ks_atk, _)) in enumerate(splits):
        jobs.append(
            delayed(_cv_single_fold)(
                i, model, util.compute_pi,
                X_prof, y_prof, pts_prof, ks_prof,
                X_atk, pts_atk, ks_atk, verbose=verbose
            )
        )

    return _results_to_df(Parallel(
        n_jobs=n_jobs, backend='loky',
        verbose=1 if verbose else 0,
    )(jobs))


def eval_model(
    seed, model, X_prof, y_prof, pts_prof, ks_prof,
    X_atk, pts_atk, ks_atk, n_repeats=1, n_jobs=-1, verbose=True
):
    rng = np.random.default_rng(seed)
    jobs = []

    for i in range(n_repeats):
        idx = rng.permutation(X_atk.shape[0])
        X_atk_shuffled = X_atk[idx]
        pts_atk_shuffled = pts_atk[idx]
        ks_atk_shuffled = ks_atk[idx]

        jobs.append(
            delayed(_cv_single_fold)(
                i, model, util.compute_pi,
                X_prof, y_prof, pts_prof, ks_prof,
                X_atk_shuffled, pts_atk_shuffled,
                ks_atk_shuffled, verbose=verbose
            )
        )

    return _results_to_df(Parallel(
        n_jobs=n_jobs, backend='loky',
        verbose=1 if verbose else 0,
    )(jobs))


def eval_and_plot(seed, model, eval_fn=eval_model, n_repeats=50, atk_data='data/raw/ASCADv.h5',
                  prof_data='data/processed/ASCADv_clean.h5'):
    X_atk, _, pts_atk, ks_atk = load_data(atk_data, as_numpy=True, attack=True)
    kw = {'seed': seed, 'model': model, 'n_repeats': n_repeats,
          'X_atk': X_atk, 'pts_atk': pts_atk, 'ks_atk': ks_atk}

    if prof_data is not None:
        X, y, *_ = load_data(prof_data, as_numpy=True, attack=False)
        kw['X'], kw['y'] = X, y

    means, stds, frs = eval_fn(**kw)
    fig = p.plot_pge(means, frs, stds)
    return means, stds, frs, fig
