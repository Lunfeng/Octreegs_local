"""Robust quantile helpers used by the TTF pruning pipeline."""
from __future__ import annotations

from typing import Iterable, Tuple

import torch


def _as_tensor(values: torch.Tensor | Iterable[float]) -> torch.Tensor:
    """Convert ``values`` into a 1D float tensor."""

    if isinstance(values, torch.Tensor):
        return values.flatten()
    return torch.as_tensor(list(values), dtype=torch.float32)


def robust_quantile(values: torch.Tensor | Iterable[float], q: float) -> torch.Tensor:
    """Return the ``q``-quantile of ``values``.

    Parameters
    ----------
    values:
        Tensor/iterable broadcast to ``torch.float32`` and flattened internally.
    q:
        Quantile in ``[0, 1]``.

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the requested quantile. Returns ``0`` when
        ``values`` is empty.
    """

    tensor = _as_tensor(values)
    if tensor.numel() == 0:
        return torch.zeros((), dtype=tensor.dtype, device=tensor.device)
    q = float(q)
    q = min(max(q, 0.0), 1.0)
    return torch.quantile(tensor, q)


def mixed_quantiles(
    alpha: torch.Tensor,
    grad: torch.Tensor,
    q_alpha: float,
    q_grad: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return quantiles for opacity/gradient tensors.

    Parameters
    ----------
    alpha / grad:
        1D float tensors containing opacity and gradient scores with shape ``[N]``.
    q_alpha / q_grad:
        Quantile levels in ``[0, 1]``.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Pair ``(Qa, Qg)`` containing the requested quantiles as scalar tensors.
    """

    Qa = robust_quantile(alpha, q_alpha)
    Qg = robust_quantile(grad, q_grad)
    return Qa, Qg


def _smoke_test() -> None:
    """Ensure quantile helpers execute."""

    alpha = torch.tensor([0.2, 0.5, 0.7])
    grad = torch.tensor([0.1, 0.4, 0.8])
    Qa, Qg = mixed_quantiles(alpha, grad, 0.5, 0.5)
    assert Qa.ndim == 0 and Qg.ndim == 0


if __name__ == "__main__":
    _smoke_test()
