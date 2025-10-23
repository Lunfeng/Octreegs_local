from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def sobel_heat(rgb: torch.Tensor) -> torch.Tensor:
    """Compute a [0, 1] edge heat map using Sobel filters."""

    if rgb.ndim != 3:
        raise ValueError("Input must be HxWx3 tensor")
    rgb = rgb.permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
    rgb = rgb.to(torch.float32)
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=rgb.device)
    sobel_y = sobel_x.t()
    sobel_x = sobel_x.expand(3, 1, 3, 3)
    sobel_y = sobel_y.expand(3, 1, 3, 3)
    grad_x = F.conv2d(rgb, sobel_x, groups=3, padding=1)
    grad_y = F.conv2d(rgb, sobel_y, groups=3, padding=1)
    grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2)).sum(1)
    grad_mag = grad_mag.squeeze(0)
    grad_mag = grad_mag - grad_mag.min()
    if grad_mag.max() > 0:
        grad_mag = grad_mag / grad_mag.max()
    return grad_mag


def topk_mask(heat: torch.Tensor, k_percent: float) -> torch.BoolTensor:
    if heat.ndim != 2:
        raise ValueError("Heat map must be HxW")
    k_percent = max(min(k_percent, 1.0), 0.0)
    if k_percent <= 0.0:
        return torch.zeros_like(heat, dtype=torch.bool)
    flat = heat.flatten()
    k = max(int(flat.numel() * k_percent), 1)
    threshold = torch.topk(flat, k).values.min()
    return heat >= threshold


def pixels_to_gaussians(high_edge_mask: torch.BoolTensor, pix2gauss: Optional[torch.Tensor]) -> torch.BoolTensor:
    """Map high edge pixels to gaussian indices.

    Parameters
    ----------
    high_edge_mask: torch.BoolTensor
        Mask over pixels (HxW).
    pix2gauss: Optional[torch.Tensor]
        Sparse mapping returned by the renderer.  Expected to be of shape
        ``(P, K)`` where each row enumerates gaussian indices contributing to a
        pixel.  When ``None`` we fallback to an all False mask.
    """

    if pix2gauss is None or pix2gauss.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=high_edge_mask.device)
    if high_edge_mask.ndim != 2:
        raise ValueError("Edge mask must be HxW")
    active_pixels = high_edge_mask.flatten().nonzero().flatten()
    if active_pixels.numel() == 0:
        return torch.zeros(pix2gauss.shape[-1], dtype=torch.bool, device=high_edge_mask.device)
    if pix2gauss.shape[0] != high_edge_mask.numel():
        raise ValueError("Pixel-to-gaussian mapping incompatible with mask")
    gauss_mask = torch.zeros(pix2gauss.shape[-1], dtype=torch.bool, device=high_edge_mask.device)
    selected = pix2gauss[active_pixels]
    gauss_indices = selected.reshape(-1)
    gauss_indices = gauss_indices[gauss_indices >= 0].unique()
    if gauss_indices.numel() > 0:
        gauss_mask[gauss_indices] = True
    return gauss_mask

