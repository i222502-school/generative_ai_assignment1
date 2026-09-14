"""Task 6: teacher-forcing training loop with gradient clipping, checkpointing, early stopping."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from q2_nmt_rnn.src.config import PAD_IDX


@dataclass(slots=True)
class EarlyStopping:
    patience: int
    best_loss: float = field(default=float("inf"), init=False)
    bad_epochs: int = field(default=0, init=False)
    improved: bool = field(default=False, init=False)

    def step(self, val_loss: float) -> bool:
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
    grad_clip_norm: float,
) -> float:
    """Token-weighted average loss -- batches have different padded lengths, so a plain
    per-batch mean would over-count short sequences relative to long ones.
    """
    model.train(optimizer is not None)
    total_loss, total_tokens = 0.0, 0

    with torch.enable_grad() if optimizer is not None else torch.no_grad():
        for batch in loader:
            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)

            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            n_tokens = int((tgt_out != PAD_IDX).sum().item())
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    return total_loss / total_tokens


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float,
    max_epochs: int,
    patience: int,
    device: torch.device,
    grad_clip_norm: float = 1.0,
    checkpoint_path: Path | None = None,
) -> FitResult:
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopping(patience=patience)

    rows: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer, grad_clip_norm)
        val_loss = _run_epoch(model, val_loader, criterion, device, None, grad_clip_norm)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_ppl": math.exp(min(train_loss, 20.0)),
                "val_loss": val_loss,
                "val_ppl": math.exp(min(val_loss, 20.0)),
            }
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
