"""Inverse-frequency class weights to counter the ~3.5:1 pneumonia:normal imbalance."""

from __future__ import annotations

import pandas as pd
import torch

from q1_cnn_xray.src.config import CLASS_NAMES


def class_weight_tensor(train_manifest: pd.DataFrame) -> torch.Tensor:
    counts = train_manifest["label"].value_counts()
    freqs = torch.tensor([counts.get(name, 0) for name in CLASS_NAMES], dtype=torch.float32)
    return freqs.sum() / (len(CLASS_NAMES) * freqs)
