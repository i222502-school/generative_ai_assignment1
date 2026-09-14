"""Train/eval transform pipelines. Augmentation is train-only, per the assignment spec.

The custom CNN consumes native grayscale input normalized with dataset-computed
statistics; the pretrained baselines need 3-channel input normalized with the
ImageNet statistics they were trained on -- see [[project_generative_ai_assignment1]].
"""

from __future__ import annotations

from enum import Enum

import torch
import torchvision.transforms.v2 as T

from q1_cnn_xray.src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


class AugmentationPolicy(str, Enum):
    LIGHT = "light"
    HEAVY = "heavy"


def _augmentation_steps(policy: AugmentationPolicy) -> list[T.Transform]:
    if policy is AugmentationPolicy.LIGHT:
        return [T.RandomHorizontalFlip(p=0.5), T.RandomRotation(degrees=7)]
    return [
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),  # zoom
        T.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # shift
        T.ColorJitter(brightness=0.2),
    ]


def _build(
    *, train: bool, channels: int, mean: tuple[float, ...], std: tuple[float, ...], policy: AugmentationPolicy
) -> T.Compose:
    steps: list[T.Transform] = [T.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        steps += _augmentation_steps(policy)
    steps += [
        T.Grayscale(num_output_channels=channels),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=mean, std=std),
    ]
    return T.Compose(steps)


def custom_cnn_transform(
    *, train: bool, mean: float, std: float, policy: AugmentationPolicy = AugmentationPolicy.HEAVY
) -> T.Compose:
    return _build(train=train, channels=1, mean=(mean,), std=(std,), policy=policy)


def pretrained_transform(*, train: bool, policy: AugmentationPolicy = AugmentationPolicy.HEAVY) -> T.Compose:
    return _build(train=train, channels=3, mean=IMAGENET_MEAN, std=IMAGENET_STD, policy=policy)
