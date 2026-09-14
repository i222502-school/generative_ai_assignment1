"""Supervised training loop: weighted cross-entropy, checkpointing, early stopping."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(slots=True)
class EarlyStopping:
    patience: int
    best_loss: float = field(default=float("inf"), init=False)
    bad_epochs: int = field(default=0, init=False)
    improved: bool = field(default=False, init=False)

    def step(self, val_loss: float) -> bool:
        """Updates state for the epoch; returns True once patience is exhausted."""
        self.improved = val_loss < self.best_loss
        if self.improved:
            self.best_loss = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


@dataclass(slots=True)
class FitResult:
    history: pd.DataFrame
    best_val_loss: float
    epochs_trained: int
    checkpoint_path: Path | None


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    l1_lambda: float,
) -> tuple[float, float]:
    model.train(optimizer is not None)
    total_loss, correct, seen = 0.0, 0, 0

    with torch.enable_grad() if optimizer is not None else torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None and l1_lambda > 0:
                loss = loss + l1_lambda * sum(p.abs().sum() for p in model.parameters())

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += images.size(0)

    return total_loss / seen, correct / seen


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float,
    max_epochs: int,
    patience: int,
    device: torch.device,
    l1_lambda: float = 0.0,
    weight_decay: float = 0.0,
    class_weights: torch.Tensor | None = None,
    checkpoint_path: Path | None = None,
) -> FitResult:
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    trainable = (p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    stopper = EarlyStopping(patience=patience)

    rows: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, optimizer, l1_lambda)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device, None, 0.0)
        rows.append(
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
        )

        should_stop = stopper.step(val_loss)
        if stopper.improved and checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
        if should_stop:
            break

    return FitResult(
        history=pd.DataFrame(rows),
        best_val_loss=stopper.best_loss,
        epochs_trained=int(rows[-1]["epoch"]),
        checkpoint_path=checkpoint_path if checkpoint_path and checkpoint_path.exists() else None,
    )
