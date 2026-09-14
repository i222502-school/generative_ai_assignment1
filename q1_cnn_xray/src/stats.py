"""Dataset-level grayscale normalization statistics for the custom CNN."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from PIL import Image

from q1_cnn_xray.src.config import IMAGE_SIZE


def grayscale_mean_std(paths: Iterable[str]) -> tuple[float, float]:
    """Pixel-wise mean/std over the training split, resized to the model's input size."""
    total, total_sq, count = 0.0, 0.0, 0
    for path in paths:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float64) / 255.0
        total += arr.sum()
        total_sq += np.square(arr).sum()
        count += arr.size

    mean = total / count
    variance = total_sq / count - mean**2
    return mean, variance**0.5
