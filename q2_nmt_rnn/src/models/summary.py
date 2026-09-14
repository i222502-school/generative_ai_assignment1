"""Layer table (output shape + param count per layer) for the Task 4 report writeup."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import torch
from torch import nn


def _output_shape(output: torch.Tensor | tuple) -> tuple[int, ...]:
    tensor = output[0] if isinstance(output, tuple) else output
    return tuple(tensor.shape[1:])


def layer_table(model: nn.Module, src: torch.Tensor, tgt_in: torch.Tensor) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def record(name: str) -> Callable[[nn.Module, tuple, object], None]:
        def hook(module: nn.Module, _inputs: tuple, output: object) -> None:
            rows.append(
                {
                    "layer": name,
                    "type": type(module).__name__,
                    "output_shape": _output_shape(output),
                    "params": sum(p.numel() for p in module.parameters(recurse=False)),
                }
            )

        return hook

    leaves = [(name, m) for name, m in model.named_modules() if next(m.children(), None) is None]
    handles = [module.register_forward_hook(record(name)) for name, module in leaves]
    try:
        model.eval()
        with torch.no_grad():
            model(src, tgt_in)
    finally:
        for handle in handles:
            handle.remove()

    return pd.DataFrame(rows)


def param_counts(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
