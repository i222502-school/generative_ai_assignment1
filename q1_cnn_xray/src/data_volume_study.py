"""Task 7's second half: retrain the primary CNN at 25/50/75/100% of the train split."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader

from q1_cnn_xray.src.class_balance import class_weight_tensor
from q1_cnn_xray.src.dataset import ChestXrayDataset
from q1_cnn_xray.src.evaluate import evaluate_model
from q1_cnn_xray.src.models.custom_cnn import PneumoniaCNN
from q1_cnn_xray.src.training.engine import fit
from q1_cnn_xray.src.transforms import custom_cnn_transform

FRACTIONS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)


def run_data_volume_study(
    manifests: dict[str, pd.DataFrame],
    mean: float,
    std: float,
    device: torch.device,
    *,
    lr: float = 1e-3,
    max_epochs: int = 30,
    patience: int = 5,
    dropout: float = 0.4,
    batch_size: int = 32,
) -> pd.DataFrame:
    eval_transform = custom_cnn_transform(train=False, mean=mean, std=std)
    val_loader = DataLoader(ChestXrayDataset(manifests["val"], eval_transform), batch_size=batch_size)
    test_loader = DataLoader(ChestXrayDataset(manifests["test"], eval_transform), batch_size=batch_size)

    rows: list[dict[str, object]] = []
    for fraction in FRACTIONS:
        train_ds = ChestXrayDataset(
            manifests["train"], custom_cnn_transform(train=True, mean=mean, std=std), subset_fraction=fraction
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model = PneumoniaCNN(dropout=dropout)
        weights = class_weight_tensor(manifests["train"].sample(frac=fraction, random_state=0))
        fit(
            model,
            train_loader,
            val_loader,
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            device=device,
            class_weights=weights,
        )

        report = evaluate_model(model, test_loader, device, f"data_fraction_{fraction}")
        report["train_fraction"] = fraction
        report["train_size"] = len(train_ds)
        rows.append(report)

    return pd.DataFrame(rows)
