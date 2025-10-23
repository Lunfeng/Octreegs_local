"""Screen-space edge utilities for the TTF pruning pipeline."""
from __future__ import annotations

import torch
import torch.nn.functional as F


_SOBEL_X = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=torch.float32)
_SOBEL_Y = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], dtype=torch.float32)


def _prepare_image(rgb: torch.Tensor) -> torch.Tensor:
    """Convert ``rgb`` (``[H, W, 3]``) into batched CHW layout."""

    if rgb.dim() != 3 or rgb.size(-1) != 3:
        raise ValueError("RGB image must have shape [H, W, 3]")
    img = rgb.permute(2, 0, 1).unsqueeze(0).contiguous()
    return img


def sobel_heat(rgb: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edge magnitude heat map.

    Parameters
    ----------
    rgb:
        Float tensor with shape ``[H, W, 3]``.

    Returns
    -------
    torch.Tensor
        Float tensor ``[H, W]`` on the same device as ``rgb``.
    """

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
    """Return a boolean mask for the top ``k_percent`` edge responses.

    Parameters
    ----------
    heat:
        Tensor ``[H, W]`` containing edge magnitude values.
    k_percent:
        Fraction in ``[0, 1]`` specifying the proportion of pixels kept.

    Returns
    -------
    torch.Tensor
        Boolean tensor ``[H, W]`` with ``True`` entries marking selected pixels.
    """

    if heat.dim() != 2:
        raise ValueError("Heat map must have shape [H, W]")
    k_percent = float(max(min(k_percent, 1.0), 0.0))
    numel = int(heat.numel())
    if numel == 0 or k_percent == 0.0:
        return torch.zeros_like(heat, dtype=torch.bool)
    k = max(int(numel * k_percent), 1)
    threshold = torch.topk(heat.flatten(), k).values.min()
    return heat >= threshold


def pixels_to_gaussians(
    high_mask: torch.Tensor,
    gauss_ids: torch.LongTensor,
    pix2gauss_ptr: torch.LongTensor,
    pix2gauss_idx: torch.LongTensor,
    pix_coords: torch.LongTensor,
    num_gaussians: int,
) -> torch.BoolTensor:
    """Map high-edge pixels to contributing Gaussian identifiers.

    Parameters
    ----------
    high_mask:
        Boolean tensor ``[H, W]`` marking edge-enhanced pixels.
    gauss_ids:
        Long tensor ``[K]`` of unique Gaussian ids referenced by the CSR mapping.
    pix2gauss_ptr:
        Long tensor ``[M + 1]`` representing CSR row pointers.
    pix2gauss_idx:
        Long tensor ``[nnz]`` with indices into ``gauss_ids``.
    pix_coords:
        Long tensor ``[M, 2]`` holding ``(y, x)`` pixel coordinates.
    num_gaussians:
        Total number of gaussians ``N``.

    Returns
    -------
    torch.BoolTensor
        Boolean tensor ``[N]`` where ``True`` denotes a gaussian that contributes to
        at least one high-edge pixel.
    """

    if high_mask.dim() != 2:
        raise ValueError("high_mask must have shape [H, W]")
    if pix_coords.shape[0] + 1 != pix2gauss_ptr.numel():
        raise ValueError("pix2gauss_ptr must have length M + 1")

    device = gauss_ids.device
    if num_gaussians <= 0:
        return torch.zeros(0, dtype=torch.bool, device=device)
    mask = torch.zeros(num_gaussians, dtype=torch.bool, device=device)

    M = pix_coords.shape[0]
    for row in range(M):
        y = int(pix_coords[row, 0].item())
        x = int(pix_coords[row, 1].item())
        if y < 0 or x < 0:
            continue
        if y >= high_mask.shape[0] or x >= high_mask.shape[1]:
            continue
        if not bool(high_mask[y, x]):
            continue
        start = int(pix2gauss_ptr[row].item())
        end = int(pix2gauss_ptr[row + 1].item())
        if start >= end:
            continue
        local_idx = pix2gauss_idx[start:end].to(device)
        if local_idx.numel() == 0:
            continue
        selected = gauss_ids[local_idx]
        mask[selected] = True
    return mask


def _smoke_test() -> None:
    """Lightweight test for edge helpers."""

    rgb = torch.rand(4, 4, 3)
    heat = sobel_heat(rgb)
    mask = topk_mask(heat, 0.25)
    gauss_ids = torch.arange(5, dtype=torch.long)
    pix_coords = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.long)
    pix2gauss_ptr = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    pix2gauss_idx = torch.tensor([0, 2, 4], dtype=torch.long)
    high_mask = torch.zeros_like(mask, dtype=torch.bool)
    high_mask[0, 0] = True
    high_mask[2, 2] = True
    gauss_mask = pixels_to_gaussians(
        high_mask,
        gauss_ids,
        pix2gauss_ptr,
        pix2gauss_idx,
        pix_coords,
        num_gaussians=5,
    )
    assert gauss_mask.sum().item() == 2 and gauss_mask[0] and gauss_mask[4]


if __name__ == "__main__":
    _smoke_test()
