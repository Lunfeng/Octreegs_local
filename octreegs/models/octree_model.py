"""High-level Octree Gaussian Splatting model wrapper with pruning helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


@dataclass
class LeafStatistics:
    depth_mean: torch.Tensor
    depth_var: torch.Tensor
    normal_mean: torch.Tensor
    normal_var: torch.Tensor
    momentum: float = 0.9

    def update(self, depth: torch.Tensor, normal: torch.Tensor, leaf_ids: torch.Tensor) -> None:
        if leaf_ids.numel() == 0:
            return
        depth = depth.detach()
        normal = normal.detach()
        for leaf in torch.unique(leaf_ids):
            mask = leaf_ids == leaf
            if mask.sum() == 0:
                continue
            d = depth[mask].mean()
            n = normal[mask].mean(dim=0)
            dv = depth[mask].var(unbiased=False)
            nv = normal[mask].var(dim=0, unbiased=False).mean()
            idx = int(leaf.item())
            self.depth_mean[idx] = self.momentum * self.depth_mean[idx] + (1 - self.momentum) * d
            self.normal_mean[idx] = self.momentum * self.normal_mean[idx] + (1 - self.momentum) * n.norm()
            self.depth_var[idx] = self.momentum * self.depth_var[idx] + (1 - self.momentum) * dv
            self.normal_var[idx] = self.momentum * self.normal_var[idx] + (1 - self.momentum) * nv

    def high_variance_mask(self, depth_q: float, normal_q: float) -> torch.Tensor:
        if self.depth_var.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.depth_var.device)
        depth_thr = torch.quantile(self.depth_var, depth_q)
        normal_thr = torch.quantile(self.normal_var, normal_q)
        return (self.depth_var >= depth_thr) | (self.normal_var >= normal_thr)


class OctreeGSModel(nn.Module):
    """Wrapper exposing pruning friendly APIs on top of the raw Gaussian data.

    Parameters
    ----------
    gaussians:
        Dictionary mapping parameter names to tensors whose leading dimension is
        the gaussian count ``[N, ...]``. Tensors are cloned into
        :class:`~torch.nn.Parameter` objects.
    leaf_ids:
        Tensor of shape ``[N]`` containing the leaf index for each gaussian.
    leaf_stats:
        Optional :class:`LeafStatistics` tracking per-leaf running statistics
        (depth/normal mean and variance). When ``None`` a zero-initialised
        tracker is created.
    """

    def __init__(
        self,
        gaussians: Dict[str, torch.Tensor],
        leaf_ids: torch.Tensor,
        leaf_stats: Optional[LeafStatistics] = None,
    ) -> None:
        super().__init__()
        if not gaussians:
            raise ValueError("Gaussian parameter dictionary cannot be empty")
        n = next(iter(gaussians.values())).shape[0]
        self._num_gaussians = n
        self.gaussians = nn.ParameterDict()
        for name, tensor in gaussians.items():
            if tensor.shape[0] != n:
                raise ValueError("All gaussian tensors must share the first dimension")
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor)
            param = nn.Parameter(tensor.clone())
            self.gaussians[name] = param
        self.register_buffer("_leaf_ids", leaf_ids.clone().long())
        self.register_buffer(
            "_prune_mask",
            torch.zeros(n, dtype=torch.bool, device=leaf_ids.device),
        )
        self._alive_indices = torch.arange(n, device=leaf_ids.device, dtype=torch.long)
        self._frozen = False
        if leaf_stats is None:
            num_leaves = int(leaf_ids.max().item()) + 1 if leaf_ids.numel() else 0
            zeros = torch.zeros(num_leaves, device=leaf_ids.device)
            leaf_stats = LeafStatistics(
                depth_mean=zeros.clone(),
                depth_var=zeros.clone(),
                normal_mean=zeros.clone(),
                normal_var=zeros.clone(),
            )
        self.leaf_stats = leaf_stats

    # ------------------------------------------------------------------
    # Exposed API
    # ------------------------------------------------------------------
    def num_gaussians(self) -> int:
        """Return the current gaussian count ``N``.

        The call performs no tensor cloning and is therefore cheap.
        """

        return int(self._num_gaussians)

    @property
    def leaf_ids(self) -> torch.LongTensor:
        """Return the gaussian-to-leaf mapping tensor.

        Returns
        -------
        torch.LongTensor
            Tensor of shape ``[N]`` on the model device containing the leaf
            index for every gaussian. The tensor always stays in sync with the
            underlying parameter pack, including after hard pruning.
        """

        return self._leaf_ids

    def set_prune_mask(self, mask: torch.BoolTensor) -> None:
        """Apply a soft pruning mask.

        Parameters
        ----------
        mask:
            Boolean tensor of shape ``[N]`` where ``True`` marks a gaussian as
            pruned. The mask is copied to internal buffers and used to derive a
            packed index that downstream renderers can leverage to gather the
            alive gaussians without per-splat branching.
        """

        if mask.shape != self._prune_mask.shape:
            raise ValueError("Mask shape mismatch")
        mask = mask.to(device=self._prune_mask.device, dtype=torch.bool)
        self._prune_mask.copy_(mask)
        self._alive_indices = torch.nonzero(~self._prune_mask, as_tuple=False).flatten()
        if self._alive_indices.numel() == 0:
            self._alive_indices = torch.zeros(0, device=self._prune_mask.device, dtype=torch.long)

    def finalize_prune(self, mask: torch.BoolTensor) -> None:
        """Hard-delete gaussians marked by ``mask`` and shrink parameter packs.

        Parameters
        ----------
        mask:
            Boolean tensor of shape ``[N]`` where ``True`` indicates gaussians to
            remove permanently. After this call all parameter tensors are
            physically compacted and :meth:`forward` implementations should no
            longer consult the soft mask.
        """

        if mask.shape != self._prune_mask.shape:
            raise ValueError("Mask shape mismatch")
        mask = mask.to(device=self._prune_mask.device, dtype=torch.bool)
        keep = ~mask
        if bool(keep.all()):
            return
        for name, param in list(self.gaussians.items()):
            kept = param.detach()[keep].contiguous()
            self.gaussians[name] = nn.Parameter(kept)
        new_leaf_ids = self._leaf_ids[keep]
        if new_leaf_ids.numel() == 0:
            remapped_leaf_ids = torch.zeros(0, device=new_leaf_ids.device, dtype=torch.long)
            num_leaves = 0
        else:
            unique, inverse = torch.unique(new_leaf_ids, sorted=True, return_inverse=True)
            remapped_leaf_ids = inverse.to(dtype=torch.long)
            num_leaves = unique.numel()
            self.leaf_stats.depth_mean = self._gather_or_zeros(
                self.leaf_stats.depth_mean, unique, num_leaves
            )
            self.leaf_stats.depth_var = self._gather_or_zeros(
                self.leaf_stats.depth_var, unique, num_leaves
            )
            self.leaf_stats.normal_mean = self._gather_or_zeros(
                self.leaf_stats.normal_mean, unique, num_leaves
            )
            self.leaf_stats.normal_var = self._gather_or_zeros(
                self.leaf_stats.normal_var, unique, num_leaves
            )
        if num_leaves == 0:
            device = self._leaf_ids.device
            self.leaf_stats.depth_mean = torch.zeros(0, device=device)
            self.leaf_stats.depth_var = torch.zeros(0, device=device)
            self.leaf_stats.normal_mean = torch.zeros(0, device=device)
            self.leaf_stats.normal_var = torch.zeros(0, device=device)
        self._leaf_ids = remapped_leaf_ids
        new_count = int(remapped_leaf_ids.numel())
        new_mask = torch.zeros(new_count, dtype=torch.bool, device=self._prune_mask.device)
        self._prune_mask = new_mask
        self._num_gaussians = new_count
        self._alive_indices = torch.arange(new_count, device=self._prune_mask.device, dtype=torch.long)

    def freeze_growth(self, flag: bool) -> None:
        self._frozen = bool(flag)

    def is_growth_frozen(self) -> bool:
        return self._frozen

    def leaf_high_variance_mask(self, leaf_ids: torch.LongTensor) -> torch.BoolTensor:
        """Identify leaves exhibiting high geometric variance.

        Parameters
        ----------
        leaf_ids:
            Tensor of shape ``[N]`` describing the leaf index of each gaussian.

        Returns
        -------
        torch.BoolTensor
            Boolean tensor of shape ``[num_leaves]`` on the same device as
            ``leaf_ids``. A value of ``True`` marks leaves whose stored depth or
            normal variance exceeds the global 0.75 quantile. When statistics are
            unavailable a zero tensor is returned.
        """

        if leaf_ids.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=leaf_ids.device)
        num_leaves = int(leaf_ids.max().item()) + 1
        stats = self.leaf_stats
        if (
            stats.depth_var.numel() < num_leaves
            or stats.normal_var.numel() < num_leaves
            or stats.depth_var.numel() == 0
        ):
            return torch.zeros(num_leaves, dtype=torch.bool, device=leaf_ids.device)
        depth_var = stats.depth_var[:num_leaves].to(device=leaf_ids.device)
        normal_var = stats.normal_var[:num_leaves].to(device=leaf_ids.device)
        depth_thr = torch.quantile(depth_var, 0.75)
        normal_thr = torch.quantile(normal_var, 0.75)
        return (depth_var >= depth_thr) | (normal_var >= normal_thr)

    # ------------------------------------------------------------------
    # Helpers for statistics updates
    # ------------------------------------------------------------------
    def update_leaf_statistics(self, depth: torch.Tensor, normal: torch.Tensor, leaf_ids: torch.Tensor) -> None:
        if not isinstance(self.leaf_stats, LeafStatistics):
            return
        self.leaf_stats.update(depth, normal, leaf_ids)

    @property
    def prune_mask(self) -> torch.BoolTensor:
        return self._prune_mask

    def alive_mask(self) -> torch.BoolTensor:
        return ~self._prune_mask

    @property
    def alive_indices(self) -> torch.LongTensor:
        """Return the packed indices of unpruned gaussians."""

        return self._alive_indices

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _gather_or_zeros(
        tensor: torch.Tensor, indices: torch.Tensor, target_len: int
    ) -> torch.Tensor:
        """Gather ``tensor`` at ``indices`` or return zeros if unavailable."""

        if tensor.numel() == 0:
            return torch.zeros(target_len, device=indices.device, dtype=tensor.dtype)
        if tensor.shape[0] <= int(indices.max().item()):
            return torch.zeros(target_len, device=indices.device, dtype=tensor.dtype)
        gathered = tensor.index_select(0, indices.to(tensor.device))
        return gathered.to(indices.device)


def _smoke_test_pruning() -> None:
    """Minimal smoke test covering mask application and hard pruning."""

    device = torch.device("cpu")
    N = 100
    gaussians = {
        "alpha": torch.rand(N, 1, device=device),
        "pos": torch.rand(N, 3, device=device),
        "scale": torch.rand(N, 3, device=device),
        "sh": torch.rand(N, 16, device=device),
    }
    leaf_ids = torch.randint(0, 10, (N,), dtype=torch.long, device=device)
    model = OctreeGSModel(gaussians, leaf_ids)

    prune_mask = torch.zeros(N, dtype=torch.bool, device=device)
    prune_mask[:20] = True
    model.set_prune_mask(prune_mask)
    assert model.prune_mask.sum().item() == 20

    before = model.num_gaussians()
    model.finalize_prune(prune_mask)
    after = model.num_gaussians()

    assert after == before - 20
    assert model.leaf_ids.shape[0] == after
    for tensor in model.gaussians.values():
        assert tensor.shape[0] == after
    print("OctreeGSModel pruning smoke test passed.")


if __name__ == "__main__":
    _smoke_test_pruning()
