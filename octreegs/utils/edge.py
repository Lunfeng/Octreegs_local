"""Screen-space edge utilities for the TTF pruning pipeline."""
from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F


_SOBEL_X = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=torch.float32)
_SOBEL_Y = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], dtype=torch.float32)


def _prepare_image(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.dim() != 3 or rgb.size(-1) != 3:
        raise ValueError("RGB image must have shape [H, W, 3]")
    img = rgb.permute(2, 0, 1).unsqueeze(0).contiguous()
    return img


def sobel_heat(rgb: torch.Tensor) -> torch.Tensor:
    """Return normalised Sobel edge magnitude for an image."""

    img = _prepare_image(rgb)
    kx = _SOBEL_X.to(img.device).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(img.device).view(1, 1, 3, 3)
    gray = 0.2989 * img[:, 0:1] + 0.5870 * img[:, 1:2] + 0.1140 * img[:, 2:3]
    grad_x = F.conv2d(gray, kx, padding=1)
    grad_y = F.conv2d(gray, ky, padding=1)
    magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)
    magnitude = magnitude.squeeze(0).squeeze(0)
    magnitude = magnitude / (magnitude.max() + 1e-8)
    return magnitude


def topk_mask(heat: torch.Tensor, k_percent: float) -> torch.Tensor:
    """Return a boolean mask selecting the top ``k_percent`` heat values."""

    if heat.dim() != 2:
        raise ValueError("Heat map must have shape [H, W]")
    k_percent = max(min(k_percent, 1.0), 0.0)
    numel = heat.numel()
    if numel == 0 or k_percent == 0.0:
        return torch.zeros_like(heat, dtype=torch.bool)
    k = max(int(numel * k_percent), 1)
    threshold = torch.topk(heat.flatten(), k).values.min()
    return heat >= threshold


def pixels_to_gaussians(high_edge_mask: torch.Tensor, aux: Optional[dict]) -> torch.Tensor:
    """Map a high-edge pixel mask to a boolean Gaussian mask.

    The ``aux`` dictionary is expected to contain a ``pix2gauss`` entry mapping
    pixels to lists of Gaussian indices. The function tolerates missing data and
    will return an all-False mask if no mapping is provided.
    """

    if aux is None:
        return torch.zeros(0, dtype=torch.bool)
    mapping = aux.get("pix2gauss") or aux.get("pixel_to_gaussians")
    if mapping is None:
        return torch.zeros(0, dtype=torch.bool)
    H, W = high_edge_mask.shape
    mask_flat = high_edge_mask.flatten()
    max_index = aux.get("num_gaussians", 0)
    if max_index == 0 and isinstance(mapping, torch.Tensor):
        max_index = int(mapping.max().item()) + 1
    gauss_mask = torch.zeros(max_index, dtype=torch.bool, device=high_edge_mask.device)
    if isinstance(mapping, torch.Tensor):
        if mapping.dim() != 2 or mapping.size(0) != mask_flat.numel():
            raise ValueError("Dense mapping tensor must have shape [H*W, K]")
        valid_rows = mask_flat.nonzero().flatten()
        if valid_rows.numel() == 0:
            return gauss_mask
        selected = mapping[valid_rows]
        gauss_mask.scatter_(0, selected.unique(), True)
        return gauss_mask

    # Fallback to Python list mapping
    for idx, is_edge in enumerate(mask_flat.tolist()):
        if not is_edge:
            continue
        gauss_indices = mapping[idx]
        if gauss_indices is None:
            continue
        for g in gauss_indices:
            if 0 <= g < gauss_mask.numel():
                gauss_mask[g] = True
    return gauss_mask
