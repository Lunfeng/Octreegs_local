from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EmaTracker:
    """Simple exponential moving average tracker."""

    num_items: int
    momentum: float = 0.9

    def __post_init__(self) -> None:
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum should be in [0, 1)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer = torch.zeros(self.num_items, dtype=torch.float32, device=device)
        self.initialised = False

    def to(self, device: torch.device | str) -> "EmaTracker":
        self.buffer = self.buffer.to(device)
        return self

    def update(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[0] != self.num_items:
            raise ValueError("EMA update size mismatch")
        values = values.to(self.buffer.device)
        if not self.initialised:
            self.buffer.copy_(values)
            self.initialised = True
        else:
            self.buffer.mul_(self.momentum).add_(values, alpha=1.0 - self.momentum)
        return self.buffer

    def resize_(self, new_size: int, fill_value: float = 0.0) -> None:
        if new_size == self.num_items:
            return
        new_buffer = torch.full((new_size,), fill_value, dtype=self.buffer.dtype, device=self.buffer.device)
        copy_len = min(self.num_items, new_size)
        if copy_len > 0:
            new_buffer[:copy_len] = self.buffer[:copy_len]
        self.num_items = new_size
        self.buffer = new_buffer
        self.initialised = copy_len > 0

    @property
    def value(self) -> torch.Tensor:
        return self.buffer

