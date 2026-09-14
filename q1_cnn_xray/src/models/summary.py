"""Layer tables (output shape + param count per layer) for the report's Task 4/5 writeups."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import torch
from torch import nn


def layer_table(model: nn.Module, input_shape: tuple[int, int, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def record(name: str) -> Callable[[nn.Module, tuple, torch.Tensor], None]:
        def hook(module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
            rows.append(
                {
                    "layer": name,
                    "type": type(module).__name__,
                    "output_shape": tuple(output.shape[1:]),
                    "params": sum(p.numel() for p in module.parameters(recurse=False)),
                }
            )

        return hook

    leaves = [(name, m) for name, m in model.named_modules() if next(m.children(), None) is None]
    handles = [module.register_forward_hook(record(name)) for name, module in leaves]
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, *input_shape))
    finally:
        for handle in handles:
            handle.remove()

    return pd.DataFrame(rows)


def param_counts(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
