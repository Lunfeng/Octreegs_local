"""Renderer utilities with pixel-to-Gaussian contribution queries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

try:
    from gaussian_renderer import render as default_render_fn
except ImportError:  # pragma: no cover - optional dependency during docs/tests
    default_render_fn = None  # type: ignore[assignment]


@dataclass
class RenderWithContribOptions:
    """Configuration controlling the pixel sampling strategy.

    Attributes
    ----------
    sample_stride:
        Horizontal/vertical stride (in pixels) between sampled locations used to
        approximate the sparse pixel-to-Gaussian mapping. Must be ``>= 1``.
    max_samples:
        Hard cap on the number of pixels retained after strided sampling. Set to
        ``0`` to disable the cap.
    radius_scale:
        Multiplicative factor applied to the per-Gaussian rasterised radius when
        testing whether a Gaussian contributes to a sampled pixel.
    chunk_size:
        Number of pixels processed per chunk when building the CSR structure.
    """

    sample_stride: int = 4
    max_samples: int = 2048
    radius_scale: float = 1.5
    chunk_size: int = 64


class Renderer:
    """Renderer wrapper exposing contribution-aware inference helpers.

    Parameters
    ----------
    model:
        Underlying Gaussian model instance accepted by the ``render_fn``.
    pipeline:
        Pipeline configuration passed through to ``render_fn``.
    background:
        Background colour tensor forwarded to ``render_fn``.
    render_fn:
        Callable implementing the base rendering logic. Defaults to
        :func:`gaussian_renderer.render` when available.
    options:
        Sampling and CSR construction options. When ``None`` the defaults from
        :class:`RenderWithContribOptions` are used.
    device:
        Optional device override applied to intermediate tensors when the base
        render output does not provide one (primarily for CPU unit tests).
    """

    def __init__(
        self,
        model,
        pipeline,
        background,
        *,
        render_fn: Optional[Callable] = None,
        options: Optional[RenderWithContribOptions] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.pipeline = pipeline
        self.background = background
        self._render_fn: Optional[Callable] = render_fn or default_render_fn
        self.options = options or RenderWithContribOptions()
        self._device_override = device

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def render(self, view, **kwargs) -> Dict[str, torch.Tensor]:
        """Render the provided view using the configured backend."""

        if self._render_fn is None:
            raise RuntimeError("No rendering function available")
        return self._render_fn(view, self.model, self.pipeline, self.background, **kwargs)

    def render_with_contrib(self, view, **kwargs) -> Dict[str, torch.Tensor]:
        """Render ``view`` and gather a sparse pixel→Gaussian mapping.

        Parameters
        ----------
        view:
            Camera/view object accepted by :meth:`render`.
        **kwargs:
            Additional keyword arguments forwarded to :meth:`render`.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:

            ``rgb``: ``FloatTensor[H, W, 3]``
                Rendered RGB image in ``[0, 1]``.
            ``gauss_ids``: ``LongTensor[K]``
                Unique Gaussian indices contributing to the pass.
            ``pix2gauss_ptr``: ``LongTensor[M+1]``
                CSR-style prefix-sum pointer for the sampled ``M`` pixels.
            ``pix2gauss_idx``: ``LongTensor[nnz]``
                Concatenated indices into ``gauss_ids`` describing
                contributions per pixel.
            ``pix_coords``: ``LongTensor[M, 2]``
                ``(y, x)`` coordinates for sampled pixels.
        """

        render_pkg = self.render(view, **kwargs)
        rgb = render_pkg["render"]
        with torch.no_grad():
            gauss_ids, ptr, idx, pix_coords = self._build_pix2gauss_map(view, render_pkg, rgb.device)
        return {
            "rgb": rgb,
            "gauss_ids": gauss_ids,
            "pix2gauss_ptr": ptr,
            "pix2gauss_idx": idx,
            "pix_coords": pix_coords,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_pix2gauss_map(
        self,
        view,
        render_pkg: Dict[str, torch.Tensor],
        rgb_device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct CSR buffers linking sampled pixels to Gaussian indices."""

        device = rgb_device if rgb_device is not None else self._device_override
        if device is None:
            device = torch.device("cpu")

        visibility = render_pkg.get("visibility_filter")
        if visibility is None:
            radii = render_pkg["radii"]
            visibility = radii > 0
        else:
            visibility = visibility.to(dtype=torch.bool)
        visible_ids = visibility.nonzero(as_tuple=False).flatten()
        if visible_ids.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, torch.zeros(1, dtype=torch.long, device=device), empty, empty.view(0, 2)

        radii = render_pkg["radii"][visible_ids]
        viewspace = render_pkg["viewspace_points"][visible_ids, :2]

        height = int(getattr(view, "image_height"))
        width = int(getattr(view, "image_width"))
        centers_yx = self._screen_to_pixel_yx(viewspace, height, width, device)
        gauss_ids = visible_ids.to(device=device, dtype=torch.long)

        pix_coords = self._sample_pixel_coords(height, width, device)
        if pix_coords.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return gauss_ids, torch.zeros(1, dtype=torch.long, device=device), empty, pix_coords

        radius_sq = (radii.to(device=device) * float(self.options.radius_scale)).pow(2)
        chunk = max(int(self.options.chunk_size), 1)

        ptr_values: List[int] = [0]
        collected: List[torch.Tensor] = []
        for pix_chunk in pix_coords.split(chunk):
            pix_float = pix_chunk.to(device=device, dtype=centers_yx.dtype)
            diff = centers_yx.unsqueeze(0) - pix_float.unsqueeze(1)
            dist_sq = diff.pow(2).sum(-1)
            mask = dist_sq <= radius_sq.unsqueeze(0)
            for row in mask:
                selected = gauss_ids[row]
                collected.append(selected)
                ptr_values.append(ptr_values[-1] + int(selected.numel()))

        if len(ptr_values) != pix_coords.size(0) + 1:
            raise RuntimeError("CSR pointer size mismatch; chunk processing bug")

        if collected:
            pix2gauss_idx = torch.cat(collected)
        else:
            pix2gauss_idx = torch.empty(0, dtype=torch.long, device=device)

        pix2gauss_ptr = torch.tensor(ptr_values, dtype=torch.long, device=device)
        return gauss_ids, pix2gauss_ptr, pix2gauss_idx, pix_coords

    def _sample_pixel_coords(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Return ``LongTensor[M, 2]`` sampling the image on a regular grid."""

        stride = max(int(self.options.sample_stride), 1)
        ys = torch.arange(0, height, stride, device=device)
        xs = torch.arange(0, width, stride, device=device)
        if ys.numel() == 0 or xs.numel() == 0:
            return torch.empty(0, 2, dtype=torch.long, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=1)
        max_samples = int(self.options.max_samples)
        if max_samples > 0 and coords.size(0) > max_samples:
            perm = torch.randperm(coords.size(0), device=device)[:max_samples]
            coords = coords[perm]
        return coords.to(dtype=torch.long)

    @staticmethod
    def _screen_to_pixel_yx(points: torch.Tensor, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Map normalised screen coordinates to ``(y, x)`` pixel positions."""

        if points.numel() == 0:
            return torch.empty(0, 2, dtype=torch.float32, device=device)
        x = (points[:, 0] * 0.5 + 0.5) * max(width - 1, 1)
        y = (points[:, 1] * 0.5 + 0.5) * max(height - 1, 1)
        stacked = torch.stack([y, x], dim=1)
        return stacked.to(device=device)


def _smoke_test() -> None:
    """Quick verification ensuring CSR assembly behaves as expected."""

    class _DummyView:
        image_height = 8
        image_width = 8

    class _DummyRenderer(Renderer):
        def __init__(self) -> None:
            super().__init__(None, None, None, render_fn=None, options=RenderWithContribOptions(sample_stride=2, max_samples=0))

        def render(self, view, **kwargs):  # type: ignore[override]
            rgb = torch.rand(view.image_height, view.image_width, 3)
            data = {
                "render": rgb,
                "visibility_filter": torch.tensor([True, False, True, True]),
                "radii": torch.tensor([2.0, 0.0, 1.0, 1.5]),
                "viewspace_points": torch.tensor(
                    [
                        [0.0, 0.0, 1.0],
                        [0.5, 0.5, 1.0],
                        [-0.5, -0.5, 1.0],
                        [0.25, -0.25, 1.0],
                    ],
                    dtype=torch.float32,
                ),
            }
            return data

    renderer = _DummyRenderer()
    view = _DummyView()
    result = renderer.render_with_contrib(view)
    assert result["rgb"].shape == (8, 8, 3)
    assert result["gauss_ids"].dtype == torch.long
    assert result["pix2gauss_ptr"].size(0) == result["pix_coords"].size(0) + 1


if __name__ == "__main__":
    _smoke_test()
