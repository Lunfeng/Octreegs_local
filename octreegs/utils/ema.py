"""Exponential moving average utilities for TTF pruning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class EmaTracker:
    """Maintain an exponential moving average for a fixed number of items.

    Parameters
    ----------
    num_items:
        Number of tracked elements. The internal buffer has shape ``[num_items]``.
    momentum:
        Smoothing factor in ``[0, 1)``. Higher values yield slower updates.
    device:
        Optional ``torch.device`` used when initialising the EMA buffer.
    dtype:
        Optional ``torch.dtype`` for the buffer. Defaults to ``torch.float32``.
    """

    num_items: int
    momentum: float = 0.9
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        """Allocate the EMA buffer and validate arguments."""

        if not 0.0 <= float(self.momentum) < 1.0:
            raise ValueError(f"Momentum must be in [0, 1); received {self.momentum}")
        device = self.device if self.device is not None else torch.device("cpu")
        self._buffer = torch.zeros(self.num_items, dtype=self.dtype, device=device)
        self._initialised = False

    def update(self, values: torch.Tensor) -> None:
        """Update the EMA in-place.

        Parameters
        ----------
        values:
            Tensor of shape ``[num_items]`` whose ``dtype`` and ``device`` must match
            the tracker buffer.
        """

        if values.shape != self._buffer.shape:
            raise ValueError(
                f"Expected tensor with shape {tuple(self._buffer.shape)}, got {tuple(values.shape)}"
            )
        if values.device != self._buffer.device or values.dtype != self._buffer.dtype:
            raise ValueError("Input tensor must share device and dtype with EMA buffer")
        if not self._initialised:
            self._buffer.copy_(values)
            self._initialised = True
        else:
            self._buffer.mul_(self.momentum).add_(values, alpha=1.0 - self.momentum)

    def values(self) -> torch.Tensor:
        """Return the EMA buffer of shape ``[num_items]``."""

        return self._buffer


def _smoke_test() -> None:
    """Minimal smoke test ensuring updates succeed."""

    tracker = EmaTracker(3, momentum=0.5)
    data = torch.tensor([1.0, 2.0, 3.0])
    tracker.update(data)
    tracker.update(data + 1.0)
    assert tracker.values().shape == (3,)


if __name__ == "__main__":
    _smoke_test()
