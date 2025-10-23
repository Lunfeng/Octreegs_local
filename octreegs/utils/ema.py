"""Exponential moving average utilities for TTF pruning.

This module implements the :class:`EmaTracker` helper which maintains a tensor
EMA with configurable momentum. The tracker is designed to work with the
Gaussian parameter tensors used by the pruning controller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class EmaTracker:
    """Track an exponential moving average for a fixed-size tensor.

    Parameters
    ----------
    num_items:
        Number of items that the EMA should track. The tracker internally stores
        a 1D tensor of this length.
    momentum:
        Momentum value in ``[0, 1)``. Higher values result in slower updates.
    device:
        Optional device on which the EMA tensor should be allocated.
    dtype:
        Optional tensor dtype. Defaults to ``torch.float32``.
    """

    num_items: int
    momentum: float = 0.9
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(f"Momentum must be in [0, 1); got {self.momentum}")
        device = self.device if self.device is not None else torch.device("cpu")
        self._value = torch.zeros(self.num_items, dtype=self.dtype, device=device)
        self._initialized = False

    @property
    def value(self) -> torch.Tensor:
        """Return the current EMA tensor."""

        return self._value

    def update(self, new_value: torch.Tensor) -> torch.Tensor:
        """Update the EMA with ``new_value``.

        ``new_value`` is broadcast to the tracker size if necessary. The
        function returns the updated EMA tensor.
        """

        if new_value.numel() != self.num_items:
            raise ValueError(
                f"Expected tensor with {self.num_items} elements, got {new_value.numel()}"
            )
        if not self._initialized:
            self._value.copy_(new_value)
            self._initialized = True
        else:
            self._value.mul_(self.momentum).add_(new_value, alpha=1.0 - self.momentum)
        return self._value

    def reset(self, fill: float = 0.0) -> None:
        """Reset the EMA to a constant value."""

        self._value.fill_(fill)
        self._initialized = False

    def state_dict(self) -> dict:
        """Return a serialisable snapshot of the tracker."""

        return {
            "value": self._value.clone(),
            "initialized": self._initialized,
            "momentum": self.momentum,
            "num_items": self.num_items,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore tracker state."""

        self.momentum = state.get("momentum", self.momentum)
        stored = state.get("value")
        if stored is None:
            raise KeyError("EMA state does not contain 'value'")
        if stored.numel() != self.num_items:
            raise ValueError(
                f"State tensor has {stored.numel()} elements; expected {self.num_items}"
            )
        self._value.copy_(stored)
        self._initialized = bool(state.get("initialized", False))
