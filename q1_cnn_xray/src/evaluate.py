"""Task 8: confusion matrix, per-class/macro P/R/F1, AUC-ROC, and a cross-model table."""

from __future__ import annotations

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader

from q1_cnn_xray.src.config import CLASS_NAMES

POSITIVE_CLASS_INDEX = CLASS_NAMES.index("PNEUMONIA")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int], list[float]]:
    model.to(device).eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []

    for images, labels in loader:
        probs = torch.softmax(model(images.to(device)), dim=1)
        y_true += labels.tolist()
        y_pred += probs.argmax(dim=1).cpu().tolist()
        y_prob += probs[:, POSITIVE_CLASS_INDEX].cpu().tolist()

    return y_true, y_pred, y_prob


def confusion_matrix_df(y_true: list[int], y_pred: list[int]) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    return pd.DataFrame(matrix, index=[f"true_{c}" for c in CLASS_NAMES], columns=[f"pred_{c}" for c in CLASS_NAMES])


def metrics_report(y_true: list[int], y_pred: list[int], y_prob: list[float]) -> dict[str, float]:
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(CLASS_NAMES)), zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

    report: dict[str, float] = {"accuracy": accuracy, "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1}
    for i, name in enumerate(CLASS_NAMES):
        report[f"{name}_precision"] = per_class_p[i]
        report[f"{name}_recall"] = per_class_r[i]
        report[f"{name}_f1"] = per_class_f1[i]

    if len(set(y_true)) > 1:
        report["auc_roc"] = roc_auc_score(y_true, y_prob)
    return report


def roc_curve_points(y_true: list[int], y_prob: list[float]) -> pd.DataFrame:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, model_name: str) -> dict[str, float]:
    y_true, y_pred, y_prob = predict(model, loader, device)
    return {"model": model_name, **metrics_report(y_true, y_pred, y_prob)}


def comparison_table(*model_reports: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(model_reports).set_index("model")
