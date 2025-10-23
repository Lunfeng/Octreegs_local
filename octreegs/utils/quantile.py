"""Robust quantile helpers used by the TTF pruning pipeline."""
from __future__ import annotations

from typing import Iterable, Tuple

import torch


def _as_tensor(values: torch.Tensor | Iterable[float]) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values
    return torch.as_tensor(list(values), dtype=torch.float32)


def robust_quantile(values: torch.Tensor | Iterable[float], q: float, default: float = 0.0) -> torch.Tensor:
    """Compute a quantile that gracefully falls back when the input is empty."""

    tensor = _as_tensor(values)
    if tensor.numel() == 0:
        return torch.tensor(default, dtype=torch.float32, device=tensor.device if tensor.is_cuda else None)
    q = float(q)
    q = min(max(q, 0.0), 1.0)
    return torch.quantile(tensor, q)


def mixed_quantiles(alpha: torch.Tensor, grad: torch.Tensor, q_alpha: float, q_grad: float,
                    default_alpha: float = 0.0, default_grad: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return quantiles for alpha and gradient tensors."""

    Qa = robust_quantile(alpha, q_alpha, default=default_alpha)
    Qg = robust_quantile(grad, q_grad, default=default_grad)
    return Qa, Qg
