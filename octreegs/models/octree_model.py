"""OctreeGS model helpers wrapping the legacy GaussianModel implementation."""
from __future__ import annotations

from typing import Optional

import torch

from scene.gaussian_model import GaussianModel


class OctreeGSModel(GaussianModel):
    """Thin wrapper around :class:`scene.gaussian_model.GaussianModel`.

    The original project exposes the :class:`GaussianModel` class directly.  For
    the pruning pipeline we rely on a few extra helper APIs.  The base class is
    already extended to expose them, but providing an explicit subclass makes
    the integration clearer for new code paths while keeping backwards
    compatibility with the existing training scripts.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # The parent class already exposes the pruning API, we simply re-export the
    # type signatures here for clarity and potential static type checkers.
    def num_gaussians(self) -> int:  # pragma: no cover - thin wrapper
        return super().num_gaussians()

    @property
    def leaf_ids(self) -> torch.LongTensor:  # pragma: no cover - thin wrapper
        return super().leaf_ids

    def set_prune_mask(self, mask: torch.BoolTensor) -> None:  # pragma: no cover
        super().set_prune_mask(mask)

    def finalize_prune(self, mask: torch.BoolTensor) -> None:  # pragma: no cover
        super().finalize_prune(mask)

    def freeze_growth(self, flag: bool) -> None:  # pragma: no cover
        super().freeze_growth(flag)

    def leaf_high_variance_mask(self, leaf_ids: torch.LongTensor) -> torch.BoolTensor:  # pragma: no cover
        return super().leaf_high_variance_mask(leaf_ids)

