import csv
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from nn import util


class MLP(nn.Module):
    def __init__(
        self, units=256, n_layers=6, in_size=1400,
        n_classes=256, dropout=0.3, dtype=torch.float32
    ):
        super().__init__()
        layers = nn.ModuleList()
        in_units = in_size

        for _ in range(n_layers):
            layers.append(nn.Linear(in_units, units, dtype=dtype))
            layers.append(nn.BatchNorm1d(units, dtype=torch.float32))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_units = units

        layers.append(nn.Linear(units, n_classes, dtype=dtype))
        self.layers = nn.ModuleList(layers)
        self.apply(self.init)

    def init(self, module):
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.zeros_(module.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def train_one_epoch(
    model, loader, optimizer, criterion, device
):
    model.train()
    info = []

    for batch_idx, (X, y, pts, ks) in enumerate(tqdm(loader, ascii=True, leave=False)):
        logits = model(X.to(device))

        optimizer.zero_grad()
        loss = criterion(logits, y.to(device))
        loss.backward()
        optimizer.step()

        info.append({
            'logits': logits.detach().to(torch.float32),
            'pts': pts, 'ks': ks, 'loss': loss.detach().unsqueeze(0)
        })

    return info


def val_one_epoch(
    model, loader, criterion, device
):
    model.eval()
    info = []

    for batch_idx, (X, y, pts, ks) in enumerate(tqdm(loader, ascii=True, leave=False)):
        logits = model(X.to(device))
        loss = criterion(logits, y.to(device))

        info.append({
            'logits': logits.detach().to(torch.float32),
            'pts': pts, 'ks': ks, 'loss': loss.detach().unsqueeze(0)
        })

    return info


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.multiprocessing.set_start_method('spawn', force=True)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    util.init_logger()

    os.makedirs('logs', exist_ok=True)
    log_writer = csv.writer(open('logs/mlp.csv', 'w', newline=''))
    log_writer.writerow(['epoch', 'train_pi', 'val_pi'])  # , 'lr'])

    model = MLP(units=256, n_layers=6, dropout=0.3, dtype=torch.float32)
    model.to(device)

    model = torch.compile(model, mode='default')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=500)

    criterion = nn.CrossEntropyLoss().to(device)

    train_loader, val_loader = util.load_dataset(workers=8, batch_size=128, dtype=torch.float32)

    best_val_loss = float('-inf')

    for epoch in range(500):
        train_info = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        with torch.no_grad():
            val_info = val_one_epoch(model, val_loader, criterion, device)

        *train_info, train_loss = util.batch_info_to_tensors(train_info)
        train_pi = util.compute_pi(*train_info)

        *val_info, val_loss = util.batch_info_to_tensors(val_info)
        val_pi = util.compute_pi(*val_info)

        logging.info(f"[{epoch+1}]; train pi: {train_pi.item():.4f} "
                     f"val pi: {val_pi.item():.4f}; "
                     f"lr: {scheduler.get_last_lr()[0]:.2e} ")

        log_writer.writerow([
            epoch+1,
            train_pi.item(),
            val_pi.item(),
            scheduler.get_last_lr()[0],
        ])

        if val_pi > best_val_loss:
            best_val_loss = val_pi
            torch.onnx.export(
                model,
                (torch.randn(1, 1400, dtype=torch.float32).to(device),),
                'nn/mlp.onnx',
                dynamo=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_shapes=(({0: torch.export.Dim("batch", min=1)},),),
            )
            logging.info("Best model saved")

        scheduler.step()
