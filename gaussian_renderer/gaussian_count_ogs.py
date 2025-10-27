import math
from typing import Optional, Tuple

import torch

from . import generate_neural_gaussians


def _project_to_ndc(points: torch.Tensor, camera) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project 3D points to NDC space using the camera transforms."""

    if points.numel() == 0:
        return points.new_zeros((0, 3)), points.new_zeros((0, 1))

    ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=1)
    clip = points_h @ camera.full_proj_transform.t()
    w = clip[:, 3:4]
    ndc = torch.zeros_like(clip[:, :3])
    positive_w = w.squeeze(-1) > 0
    if positive_w.any():
        ndc[positive_w] = clip[positive_w, :3] / w[positive_w]
    return ndc, w


def _estimate_pixel_footprint(
    xyz: torch.Tensor,
    scaling: torch.Tensor,
    camera,
) -> torch.Tensor:
    """Approximate pixel footprint area in pixels."""

    if xyz.numel() == 0:
        return xyz.new_zeros((0,))

    camera_center = camera.camera_center.to(xyz.device)
    direction = xyz - camera_center
    depth = torch.linalg.norm(direction, dim=1).clamp(min=1e-6)

    width = float(camera.image_width)
    height = float(camera.image_height)
    fx = width / (2.0 * math.tan(float(camera.FoVx) * 0.5))
    fy = height / (2.0 * math.tan(float(camera.FoVy) * 0.5))

    scaling = scaling[:, :3].abs().clamp(min=1e-6)
    radius_x = scaling[:, 0]
    radius_y = scaling[:, 1]

    pixel_radius_x = torch.tensor(fx, device=xyz.device, dtype=xyz.dtype) * radius_x / depth
    pixel_radius_y = torch.tensor(fy, device=xyz.device, dtype=xyz.dtype) * radius_y / depth

    area = math.pi * torch.clamp(pixel_radius_x, min=0.0) * torch.clamp(pixel_radius_y, min=0.0)
    return area


def count_render_for_view(
    viewpoint_camera,
    gaussians,
    bg_color,
    tile_size: int = 16,
    return_offset2anchor: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Estimate per-offset hit counts for a camera view.

    This implementation relies on a projection-based fallback that avoids
    touching CUDA rasterization kernels. When an offset projects outside the
    frustum we report zero hits. The estimated footprint is rounded to the
    nearest integer to emulate discrete pixel counts.
    """

    del bg_color  # kept for API compatibility / future extensions
    _ = tile_size

    with torch.no_grad():
        target_device = device or gaussians.get_anchor.device
        visible_mask = getattr(gaussians, "_anchor_mask", None)

        was_training = gaussians.get_color_mlp.training
        if was_training:
            gaussians.eval()

        try:
            generated = generate_neural_gaussians(
                viewpoint_camera,
                gaussians,
                visible_mask=visible_mask,
                is_training=True,
            )
        finally:
            if was_training:
                gaussians.train()

        xyz, _color, _opacity, scaling, _rot, _neural_opacity, mask = generated

        if xyz.numel() == 0:
            empty = torch.zeros((0,), dtype=torch.int32, device=target_device)
            return empty, (empty if return_offset2anchor else None)

        xyz = xyz.to(target_device)
        scaling = scaling.to(target_device)

        ndc, clip_w = _project_to_ndc(xyz, viewpoint_camera)
        in_front = clip_w.squeeze(-1) > 0
        inside = (
            (ndc[:, 0].abs() <= 1.2)
            & (ndc[:, 1].abs() <= 1.2)
            & in_front
        )

        footprint = _estimate_pixel_footprint(xyz, scaling, viewpoint_camera)
        hit_estimate = torch.where(
            inside,
            torch.round(footprint).clamp(min=0.0),
            torch.zeros_like(footprint),
        )

        hit_count_offset = hit_estimate.to(torch.int32)

        if not return_offset2anchor:
            return hit_count_offset, None

        n_offsets = gaussians.n_offsets
        total_offsets = mask.numel()
        anchor_indices = torch.arange(total_offsets // n_offsets, device=mask.device)
        anchor_indices = anchor_indices.repeat_interleave(n_offsets)
        anchor_indices = anchor_indices[mask]
        offset2anchor = anchor_indices.to(torch.int32).to(target_device)
        return hit_count_offset.to(target_device), offset2anchor
