"""Resumable 8-axis grid search over the primary CNN's hyperparameters (Task 7).

256 full training runs (2 values x 8 axes) don't fit in one Colab session, so every
completed combo is appended to a results CSV keyed by a content hash of its
hyperparameters; re-running the module skips combos already present in that CSV.
See [[project_generative_ai_assignment1]] for why the grid is shaped this way.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from q1_cnn_xray.src.class_balance import class_weight_tensor
from q1_cnn_xray.src.config import CHECKPOINT_DIR
from q1_cnn_xray.src.dataset import ChestXrayDataset
from q1_cnn_xray.src.models.custom_cnn import NormScheme, PneumoniaCNN
from q1_cnn_xray.src.training.engine import fit
from q1_cnn_xray.src.transforms import AugmentationPolicy, custom_cnn_transform

# One row per hyperparameter's search values -- kept as plain tuples (not a
# Cartesian-friendly framework object) so the axis definitions stay readable
# as the "hyperparameter . range" half of the Task 7 report table.
BATCH_SIZES: tuple[int, ...] = (16, 32)
LEARNING_RATES: tuple[float, ...] = (1e-3, 1e-4)
MAX_EPOCHS_OPTIONS: tuple[int, ...] = (15, 30)
DROPOUTS: tuple[float, ...] = (0.3, 0.5)
PATIENCES: tuple[int, ...] = (3, 6)
REG_SETTINGS: tuple[tuple[str, float], ...] = (("none", 0.0), ("l2", 1e-4))
NORM_SCHEMES: tuple[NormScheme, ...] = (NormScheme.BATCH, NormScheme.GROUP)
AUGMENTATION_POLICIES: tuple[AugmentationPolicy, ...] = (AugmentationPolicy.LIGHT, AugmentationPolicy.HEAVY)


@dataclass(frozen=True, slots=True)
class HyperParams:
    batch_size: int
    lr: float
    max_epochs: int
    dropout: float
    patience: int
    reg_kind: str
    reg_lambda: float
    norm_scheme: NormScheme
    augmentation_policy: AugmentationPolicy

    @property
    def combo_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def full_grid() -> list[HyperParams]:
    axes = itertools.product(
        BATCH_SIZES,
        LEARNING_RATES,
        MAX_EPOCHS_OPTIONS,
        DROPOUTS,
        PATIENCES,
        REG_SETTINGS,
        NORM_SCHEMES,
        AUGMENTATION_POLICIES,
    )
    return [
        HyperParams(batch_size, lr, max_epochs, dropout, patience, reg_kind, reg_lambda, norm_scheme, aug_policy)
        for batch_size, lr, max_epochs, dropout, patience, (reg_kind, reg_lambda), norm_scheme, aug_policy in axes
    ]


def _already_run(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    return set(pd.read_csv(results_path)["combo_id"])


def _append_result(results_path: Path, row: dict[str, object]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    header = not results_path.exists()
    pd.DataFrame([row]).to_csv(results_path, mode="a", header=header, index=False)


def run_grid_search(
    manifests: dict[str, pd.DataFrame],
    mean: float,
    std: float,
    results_path: Path,
    device: torch.device,
    combos: list[HyperParams] | None = None,
) -> pd.DataFrame:
    combos = combos if combos is not None else full_grid()
    done = _already_run(results_path)
    weights = class_weight_tensor(manifests["train"])

    for hp in combos:
        if hp.combo_id in done:
            continue

        train_transform = custom_cnn_transform(train=True, mean=mean, std=std, policy=hp.augmentation_policy)
        eval_transform = custom_cnn_transform(train=False, mean=mean, std=std)
        train_loader = DataLoader(
            ChestXrayDataset(manifests["train"], train_transform), batch_size=hp.batch_size, shuffle=True
        )
        val_loader = DataLoader(ChestXrayDataset(manifests["val"], eval_transform), batch_size=hp.batch_size)

        model = PneumoniaCNN(dropout=hp.dropout, norm_scheme=hp.norm_scheme)
        result = fit(
            model,
            train_loader,
            val_loader,
            lr=hp.lr,
            max_epochs=hp.max_epochs,
            patience=hp.patience,
            device=device,
            l1_lambda=hp.reg_lambda if hp.reg_kind == "l1" else 0.0,
            weight_decay=hp.reg_lambda if hp.reg_kind == "l2" else 0.0,
            class_weights=weights,
            checkpoint_path=CHECKPOINT_DIR / "grid_search" / f"{hp.combo_id}.pth",
        )

        _append_result(
            results_path,
            {
                "combo_id": hp.combo_id,
                **asdict(hp),
                "best_val_loss": result.best_val_loss,
                "final_val_acc": result.history["val_acc"].iloc[-1],
                "epochs_trained": result.epochs_trained,
            },
        )
        done.add(hp.combo_id)

    return pd.read_csv(results_path)


def best_per_hyperparameter(results: pd.DataFrame) -> pd.DataFrame:
    """One row per hyperparameter axis: the range searched and the value at the best run's setting."""
    best_row = results.loc[results["best_val_loss"].idxmin()]
    axis_columns = [
        "batch_size", "lr", "max_epochs", "dropout", "patience", "reg_kind", "reg_lambda", "norm_scheme",
        "augmentation_policy",
    ]
    return pd.DataFrame(
        {
            "hyperparameter": axis_columns,
            "range_searched": [sorted(results[col].unique().tolist(), key=str) for col in axis_columns],
            "optimal_value": [best_row[col] for col in axis_columns],
        }
    )
