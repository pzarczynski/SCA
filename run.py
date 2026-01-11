import hydra
import mlflow
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from tqdm import tqdm

from tools import compute_guessing_entropy, load_data

mlflow.set_tracking_uri("sqlite:///mlruns.db")


@hydra.main(version_base=None, config_path="configs", config_name='config')
def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.experiment_name)
    model = instantiate(cfg.model)

    X, y, pts, ks = load_data(cfg.data.path, test=cfg.data.test)

    mlflow.start_run()
    mlflow.log_params(cfg)

    kf = KFold(n_splits=cfg.cv_splits)
    scores = []

    pbar = tqdm(total=cfg.cv_splits, ascii=True)

    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        pts_val, ks_val = pts[val_index], ks[val_index]

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)
        fold_score = compute_guessing_entropy(proba, pts_val, ks_val)

        mlflow.log_metric(f"fold_{fold}_score", fold_score)
        scores.append(fold_score)

        pbar.update(1)
        pbar.set_description(f"Fold {fold} score: {fold_score:.2f}")

    mlflow.log_metric("mean_cv_score", np.mean(scores))
    mlflow.end_run()


if __name__ == "__main__":
    main()
