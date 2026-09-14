"""The primary model: a from-scratch CNN sized for a ~5.8k grayscale X-ray dataset.

Four conv blocks double channels 32->64->128->256 while halving spatial size via
max-pool. Global average pooling replaces a flatten+dense head -- flattening the
final 256x14x14 map into a dense layer would add ~6.4M parameters on top of a
26k-image train split, which is exactly the overfitting risk the assignment's
Task 9 asks us to discuss. GAP collapses that to 256 features for ~1/25th the cost.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import torch
from torch import nn

_CHANNELS: tuple[int, ...] = (32, 64, 128, 256)
_GROUP_NORM_GROUPS = 8  # divides every channel width in _CHANNELS evenly


class NormScheme(str, Enum):
    BATCH = "batch"
    GROUP = "group"


def _group_norm(num_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_GROUP_NORM_GROUPS, num_channels)


def _norm_builder(scheme: NormScheme) -> Callable[[int], nn.Module]:
    return nn.BatchNorm2d if scheme is NormScheme.BATCH else _group_norm


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: Callable[[int], nn.Module]) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = norm(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(torch.relu(self.norm(self.conv(x))))


class PneumoniaCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.4,
        norm_scheme: NormScheme = NormScheme.BATCH,
    ) -> None:
        super().__init__()
        norm = _norm_builder(norm_scheme)
        in_channels = (1, *_CHANNELS[:-1])
        self.blocks = nn.Sequential(*(ConvBlock(c_in, c_out, norm) for c_in, c_out in zip(in_channels, _CHANNELS)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(_CHANNELS[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.blocks(x)))
