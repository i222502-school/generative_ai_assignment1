"""torch.utils.data.Dataset over the manifest CSVs produced by data.build_manifests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform

from q1_cnn_xray.src.config import CLASS_NAMES

LABEL_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


class ChestXrayDataset(Dataset):
    def __init__(self, manifest: Path | pd.DataFrame, transform: Transform, subset_fraction: float = 1.0) -> None:
        df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
        if not 0.0 < subset_fraction <= 1.0:
            raise ValueError(f"subset_fraction must be in (0, 1], got {subset_fraction}")
        if subset_fraction < 1.0:
            df = df.sample(frac=subset_fraction, random_state=0).reset_index(drop=True)

        self.paths: list[str] = df["path"].tolist()
        self.labels: list[int] = [LABEL_TO_INDEX[label] for label in df["label"]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as img:
            image = self.transform(img.convert("RGB"))
        return image, self.labels[index]
