import os
import sys

rootdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if rootdir not in sys.path:
    os.chdir(rootdir)
    sys.path.insert(0, rootdir)

import logging
import pickle

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm.auto import tqdm, trange

sns.set_theme(style="ticks", palette="dark")

from scripts import plots, util
from scripts.cv import cv, cv_precomputed, eval_model, eval_onnx

util.init_logger()

SEED = 42

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    PolynomialFeatures,
    StandardScaler,
)


def simplerf(seed, depth=15, n_estimators=300, min_samples_leaf=500, n_jobs=-1):
    return RFC(n_estimators=n_estimators, max_depth=depth,
               random_state=seed, min_samples_leaf=min_samples_leaf,
               n_jobs=n_jobs)
