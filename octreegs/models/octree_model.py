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
    """Wrapper exposing pruning friendly APIs on top of the raw Gaussian data."""

    def __init__(self, gaussians: Dict[str, torch.Tensor], leaf_ids: torch.Tensor,
                 leaf_stats: Optional[LeafStatistics] = None) -> None:
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
        self.register_buffer("_leaf_ids", leaf_ids.clone())
        self.register_buffer("_prune_mask", torch.zeros(n, dtype=torch.bool, device=leaf_ids.device))
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
        return int(self.gaussians[next(iter(self.gaussians))].shape[0])

    @property
    def leaf_ids(self) -> torch.LongTensor:
        return self._leaf_ids.long()

    def set_prune_mask(self, mask: torch.BoolTensor) -> None:
        if mask.shape != self._prune_mask.shape:
            raise ValueError("Mask shape mismatch")
        self._prune_mask.copy_(mask)

    def finalize_prune(self, mask: torch.BoolTensor) -> None:
        if mask.shape != self._prune_mask.shape:
            raise ValueError("Mask shape mismatch")
        keep = ~mask
        if keep.sum() == keep.numel():
            return
        for name, param in list(self.gaussians.items()):
            kept = param.data[keep]
            new_param = nn.Parameter(kept)
            # ParameterDict does not allow in-place assignment, so we re-register.
            self.gaussians[name] = new_param
        new_leaf_ids = self._leaf_ids[keep]
        self.register_buffer("_leaf_ids", new_leaf_ids)
        self.register_buffer("_prune_mask", torch.zeros_like(new_leaf_ids, dtype=torch.bool))
        self._num_gaussians = int(new_leaf_ids.numel())
        if new_leaf_ids.numel() == 0:
            self.leaf_stats.depth_mean = torch.zeros(0, device=new_leaf_ids.device)
            self.leaf_stats.depth_var = torch.zeros(0, device=new_leaf_ids.device)
            self.leaf_stats.normal_mean = torch.zeros(0, device=new_leaf_ids.device)
            self.leaf_stats.normal_var = torch.zeros(0, device=new_leaf_ids.device)
        else:
            num_leaves = int(new_leaf_ids.max().item()) + 1
            self.leaf_stats.depth_mean = self.leaf_stats.depth_mean[:num_leaves].clone()
            self.leaf_stats.depth_var = self.leaf_stats.depth_var[:num_leaves].clone()
            self.leaf_stats.normal_mean = self.leaf_stats.normal_mean[:num_leaves].clone()
            self.leaf_stats.normal_var = self.leaf_stats.normal_var[:num_leaves].clone()

    def freeze_growth(self, flag: bool) -> None:
        self._frozen = bool(flag)

    def is_growth_frozen(self) -> bool:
        return self._frozen

    def leaf_high_variance_mask(self, leaf_ids: Optional[torch.LongTensor] = None,
                                depth_quantile: float = 0.8,
                                normal_quantile: float = 0.8) -> torch.BoolTensor:
        if leaf_ids is None:
            leaf_ids = self._leaf_ids
        if leaf_ids.numel() == 0:
            return torch.zeros(0, dtype=torch.bool, device=self._leaf_ids.device)
        num_leaves = int(leaf_ids.max().item()) + 1
        if self.leaf_stats.depth_var.numel() != num_leaves:
            zeros = torch.zeros(num_leaves, device=self._leaf_ids.device)
            self.leaf_stats.depth_var = zeros.clone()
            self.leaf_stats.normal_var = zeros.clone()
        return self.leaf_stats.high_variance_mask(depth_quantile, normal_quantile)

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
