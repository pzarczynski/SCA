import torch
import torch.nn.functional as F
from sklearn.model_selection import RepeatedStratifiedKFold
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from sca import util
from sca.util import init_logger, load_data, transform_splits

SBOX = torch.tensor(util.SBOX, dtype=torch.long)
LOG2 = torch.log(torch.tensor(2.0))


def load_dataset(
    path='data/processed/ASCADv.h5', val_size=0.2,
    workers=1, batch_size=128, dtype=torch.float32
):
    X, y, pts, ks, _ = util.load_data(path)
    (X, y, pts, ks) = (
        torch.tensor(X, dtype=dtype),
        torch.tensor(y, dtype=torch.long),
        torch.tensor(pts, dtype=torch.long),
        torch.tensor(ks, dtype=torch.long)
    )
    ds = TensorDataset(X, y, pts, ks)

    val_len = int(len(ds) * val_size)
    train_ds, val_ds = random_split(ds, [len(ds) - val_len, val_len])

    X_train = X[train_ds.indices]
    mean = X_train.mean(dim=0)
    std  = X_train.std(dim=0).clamp(min=1e-8)
    X.sub_(mean).div_(std)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        persistent_workers=True, pin_memory=True,
        num_workers=workers
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        persistent_workers=True, pin_memory=True,
        num_workers=workers
    )

    return train_loader, val_loader


def batch_info_to_tensors(batch_info: list[dict]):
    unpacked = zip(*(b.values() for b in batch_info))
    return [torch.cat(x, dim=0) for x in unpacked]


def compute_pi(logits, pts, ks):
    logits, pts, ks = logits, pts, ks
    log2_proba = F.log_softmax(logits, dim=1) / LOG2
    n_traces, n_classes = logits.shape

    idx = torch.arange(n_traces, device=log2_proba.device)
    sbox_idx = SBOX.to(log2_proba.device)[(pts ^ ks) % 256]
    correct = log2_proba[idx, sbox_idx]

    sum_ = torch.zeros(n_classes,
                       device=log2_proba.device,
                       dtype=log2_proba.dtype)
    sum_.index_add_(0, ks.to(log2_proba.device), correct)

    den = torch.zeros(n_classes,
                      device=log2_proba.device,
                      dtype=log2_proba.dtype)
    den.index_add_(0, ks.to(log2_proba.device), torch.ones_like(correct))

    den = torch.clamp(den, min=1.0)
    return torch.mean(8.0 + (sum_ / den))


def make_splits(seed, pl, X, y, pts, ks, n_splits, n_repeats):
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )

    splits = []
    for prof_index, atk_index in tqdm(rskf.split(X, ks), ascii=True,
                                      total=n_splits * n_repeats,
                                      desc='Precomputing splits'):
        X_prof, y_prof, pts_prof, ks_prof = (
            X[prof_index], y[prof_index], pts[prof_index], ks[prof_index]
        )
        X_atk, y_atk, pts_atk, ks_atk = (
            X[atk_index], y[atk_index], pts[atk_index], ks[atk_index]
        )
        X_prof = pl.fit_transform(X_prof, y_prof)
        X_atk = pl.transform(X_atk)
        splits.append((
            (
                torch.tensor(X_prof, dtype=torch.float32),
                torch.tensor(y_prof, dtype=torch.long),
                torch.tensor(pts_prof, dtype=torch.long),
                torch.tensor(ks_prof, dtype=torch.long)
            ),
            (
                torch.tensor(X_atk, dtype=torch.float32),
                torch.tensor(y_atk, dtype=torch.long),
                torch.tensor(pts_atk, dtype=torch.long),
                torch.tensor(ks_atk, dtype=torch.long)
            ),
        ))

    return splits
