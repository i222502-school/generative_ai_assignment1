"""VGG16 / ResNet50 comparison baselines, each swappable between frozen and fine-tuned.

Frozen: only the replaced classifier head trains. Fine-tuned: the backbone's last
stage unfreezes too, at a lower LR than the head (set by the caller's optimizer
param groups) so pretrained features aren't wrecked by early large-gradient updates.
"""

from __future__ import annotations

from enum import Enum

from torch import nn
from torchvision.models import ResNet50_Weights, VGG16_Weights, resnet50, vgg16


class TrainMode(str, Enum):
    FROZEN = "frozen"
    FINE_TUNED = "fine_tuned"


def _replace_head(in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))


def build_vgg16(mode: TrainMode, num_classes: int = 2, dropout: float = 0.4) -> nn.Module:
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = _replace_head(in_features, num_classes, dropout)

    if mode is TrainMode.FINE_TUNED:
        for param in model.features[-6:].parameters():  # last conv block
            param.requires_grad = True

    return model


def build_resnet50(mode: TrainMode, num_classes: int = 2, dropout: float = 0.4) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = _replace_head(model.fc.in_features, num_classes, dropout)

    if mode is TrainMode.FINE_TUNED:
        for param in model.layer4.parameters():  # last residual stage
            param.requires_grad = True

    return model
