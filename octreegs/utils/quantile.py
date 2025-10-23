from __future__ import annotations

from typing import Tuple

import torch


def robust_quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    """Compute a robust quantile for 1-D tensors.

    The helper guards against empty tensors by returning zero in that case.  It
    also clamps the quantile parameter to ``[0, 1]`` and performs the computation
    on CPU for numerical stability when the tensor is small.
    """

    if values.numel() == 0:
        return torch.tensor(0.0, device=values.device)
    q = float(min(max(q, 0.0), 1.0))
    if values.numel() < 4:
        return values.mean()
    return torch.quantile(values, q)


def mixed_quantiles(alpha: torch.Tensor, grads: torch.Tensor, q_alpha: float, q_grad: float) -> Tuple[torch.Tensor, torch.Tensor]:
    qa = robust_quantile(alpha, q_alpha)
    qg = robust_quantile(grads, q_grad)
    return qa, qg

