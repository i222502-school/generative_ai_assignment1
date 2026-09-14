"""Task 9: surfaces the misclassified test images for the report's failure-mode discussion."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from q1_cnn_xray.src.config import CLASS_NAMES
from q1_cnn_xray.src.dataset import ChestXrayDataset
from q1_cnn_xray.src.evaluate import predict


def top_misclassified(model: nn.Module, dataset: ChestXrayDataset, device: torch.device, n: int = 10) -> pd.DataFrame:
    """Ranked by confidence in the wrong answer -- the most confidently wrong cases first.

    Takes the dataset (not a caller-built loader) so prediction order is guaranteed to
    line up with dataset.paths -- a shuffled loader would silently mis-pair the two.
    """
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    y_true, y_pred, y_prob = predict(model, loader, device)
    df = pd.DataFrame({"path": dataset.paths, "true": y_true, "pred": y_pred, "pneumonia_prob": y_prob})
    wrong = df[df["true"] != df["pred"]].copy()
    wrong["confidence"] = wrong.apply(
        lambda row: row["pneumonia_prob"] if row["pred"] == 1 else 1 - row["pneumonia_prob"], axis=1
    )
    return wrong.sort_values("confidence", ascending=False).head(n).reset_index(drop=True)


def save_misclassified_grid(cases: pd.DataFrame, out_path: Path) -> None:
    if cases.empty:
        return
    cols = 5
    rows = -(-len(cases) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    for ax, (_, case) in zip(axes.flat, cases.iterrows()):
        with Image.open(case["path"]) as img:
            ax.imshow(img.convert("L"), cmap="gray")
        ax.set_title(f"true={CLASS_NAMES[case['true']]}\npred={CLASS_NAMES[case['pred']]} ({case['confidence']:.2f})", fontsize=9)
        ax.axis("off")
    for ax in axes.flat[len(cases):]:
        ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
