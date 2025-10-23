"""Sanity checker for the Trimming-the-Fat (TTF) pruning pipeline."""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch

from octreegs.pruning.ttf_core import TtfController

LOGGER = logging.getLogger(__name__)


class _DummyModel(torch.nn.Module):
    """Minimal gaussian container compatible with :class:`TtfController`."""

    def __init__(self, num_gaussians: int, num_leaves: int) -> None:
        super().__init__()
        self.gaussians = {
            "alpha": torch.rand(num_gaussians, 1),
            "scale": torch.rand(num_gaussians, 3),
            "sh": torch.rand(num_gaussians, 16),
        }
        ids = torch.arange(num_leaves, dtype=torch.long)
        self.register_buffer(
            "_leaf_ids",
            ids.repeat_interleave(math.ceil(num_gaussians / num_leaves))[:num_gaussians],
        )
        self.register_buffer("_mask", torch.zeros(num_gaussians, dtype=torch.bool))

    def num_gaussians(self) -> int:
        """Return total gaussian count without copying tensors."""

        return int(self._leaf_ids.numel())

    @property
    def leaf_ids(self) -> torch.LongTensor:
        """Tensor mapping each gaussian to its octree leaf."""

        return self._leaf_ids

    def set_prune_mask(self, mask: torch.BoolTensor) -> None:
        """Apply a soft mask used during rendering."""

        self._mask = mask.clone()

    def finalize_prune(self, mask: torch.BoolTensor) -> None:
        """Physically remove pruned gaussians from the parameter tensors."""

        keep = ~mask
        for key in list(self.gaussians.keys()):
            self.gaussians[key] = self.gaussians[key][keep]
        self._leaf_ids = self._leaf_ids[keep]
        self._mask = torch.zeros_like(self._leaf_ids, dtype=torch.bool)

    def freeze_growth(self, flag: bool) -> None:  # pragma: no cover - interface noop
        """The dummy model never performs growth; provided for API parity."""

    def leaf_high_variance_mask(self, leaf_ids: torch.LongTensor) -> torch.BoolTensor:
        """Mark one eighth of leaves as high-variance to exercise the heuristic."""

        if leaf_ids.numel() == 0:
            return torch.zeros(0, dtype=torch.bool)
        num_leaves = int(leaf_ids.max().item()) + 1
        mask = torch.zeros(num_leaves, dtype=torch.bool)
        if num_leaves > 0:
            mask[: max(1, num_leaves // 8)] = True
        return mask


class _DummyRenderer:
    """Renderer stub returning random contribution mappings."""

    def __init__(self, num_gaussians: int) -> None:
        self.num_gaussians = num_gaussians

    def render_with_contrib(self, view) -> Dict[str, torch.Tensor]:
        """Produce a sparse pixel→gaussian mapping for testing."""

        height = width = 32
        rgb = torch.rand(height, width, 3)
        gauss_ids = torch.arange(self.num_gaussians, dtype=torch.long)
        sampled = 128
        pix_coords = torch.randint(0, height, (sampled, 2), dtype=torch.long)
        ptr = torch.arange(0, sampled + 1, dtype=torch.long)
        idx = torch.randint(0, self.num_gaussians, (sampled,), dtype=torch.long)
        return {
            "rgb": rgb,
            "gauss_ids": gauss_ids,
            "pix2gauss_ptr": ptr,
            "pix2gauss_idx": idx,
            "pix_coords": pix_coords,
        }


def _dummy_train_step(iters: int, **kwargs) -> Dict[str, int]:
    """Mimic the trainer signature while invoking optional hooks."""

    hook = kwargs.get("hook_after_backward")
    if callable(hook):
        hook()
    return {"iters": int(iters)}


def _build_dummy_controller(
    num_gaussians: int = 512,
    num_leaves: int = 32,
    seed: Optional[int] = 0,
) -> TtfController:
    """Create a controller instance configured for synthetic testing."""

    if seed is not None:
        torch.manual_seed(seed)
    model = _DummyModel(num_gaussians, num_leaves)
    renderer = _DummyRenderer(num_gaussians)
    cfg = {
        "common": {
            "lambda_dssim": 0.2,
            "ema_momentum": 0.9,
            "edge_topk_percent": 0.15,
            "edge_boost": 1.6,
            "leaf_q_reduce": 0.10,
            "nmin_per_leaf": 24,
            "rollback_enable": True,
            "rollback_ratio": 0.05,
            "rollback_window_rounds": 2,
            "local_psnr_drop": 0.2,
            "rounds": 2,
        },
        "balanced": {"gamma_iter": 0.325},
    }
    controller = TtfController(model, renderer, _dummy_train_step, _dummy_eval_step, cfg, device="cpu")
    controller._stats_ready = True  # type: ignore[attr-defined]
    controller.gamma_iter = float(cfg["balanced"]["gamma_iter"])
    controller.grad_ema.update(torch.rand(num_gaussians))
    controller.alpha_ema.update(torch.rand(num_gaussians))
    return controller


def _dummy_eval_step(return_rgb: bool = False, tiles: bool = False) -> Dict[str, torch.Tensor]:
    """Return constant evaluation metrics for the dummy controller."""

    data: Dict[str, torch.Tensor] = {
        "psnr": torch.tensor(30.0),
        "ssim": torch.tensor(0.9),
        "lpips": torch.tensor(0.05),
        "fps": torch.tensor(24.0),
    }
    if return_rgb:
        data.update(
            {
                "rgb": torch.rand(8, 8, 3),
                "gauss_ids": torch.arange(8, dtype=torch.long),
                "pix2gauss_ptr": torch.tensor([0, 2, 4], dtype=torch.long),
                "pix2gauss_idx": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                "pix_coords": torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
            }
        )
    if tiles:
        data["tile_psnr_delta"] = torch.tensor([-0.1, -0.05])
    return data


def run_random_sanity(seed: Optional[int] = 0) -> Dict[str, float]:
    """Execute a synthetic pruning round and assert basic invariants."""

    controller = _build_dummy_controller(seed=seed)
    N = controller.model.num_gaussians()
    leaf_ids = controller.model.leaf_ids
    mask_before = controller.prune_mask.clone()
    controller.prune_round(controller.round_idx + 1)
    mask_after = controller.prune_mask.clone()
    pruned = int(mask_after.sum().item())
    alive = ~mask_after
    gamma = float(controller.gamma_iter)
    target = math.ceil(N * gamma)
    if pruned > target:
        raise AssertionError(f"Pruned {pruned} gaussians exceeding target {target}")

    nmin = int(controller.common_cfg.get("nmin_per_leaf", 24))
    keep_ratio = 0.30
    violations = []
    for leaf in torch.unique(leaf_ids):
        leaf_mask = leaf_ids == leaf
        total = int(leaf_mask.sum().item())
        keep = int((alive & leaf_mask).sum().item())
        required = min(total, max(nmin, math.ceil(total * keep_ratio)))
        if keep < required:
            violations.append((int(leaf.item()), keep, required, total))
    if violations:
        raise AssertionError(f"Leaf quota violations detected: {violations[:4]}")

    LOGGER.info(
        "Random sanity: N=%d pruned=%d target<=%d leaves_ok=%s", N, pruned, target, not violations
    )
    delta = int(mask_after.sum().item() - mask_before.sum().item())
    return {
        "num_gaussians": float(N),
        "pruned": float(pruned),
        "target": float(target),
        "delta": float(delta),
    }


def _parse_metrics(path: Path) -> Dict[str, float]:
    """Load metrics stored as ``key: value`` or ``key=value`` lines."""

    metrics: Dict[str, float] = {}
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, value = line.split(":", maxsplit=1)
        elif "=" in line:
            key, value = line.split("=", maxsplit=1)
        else:
            continue
        try:
            metrics[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return metrics


def _extract_gaussians(metrics: Dict[str, float], fallback: Optional[float]) -> Optional[float]:
    """Pull gaussian count from metrics when present."""

    for key in ("gaussians", "num_gaussians", "num_gauss", "N"):
        if key in metrics:
            return float(metrics[key])
    return fallback


def _summarise_real_metrics(
    before: Dict[str, float],
    after: Dict[str, float],
    gauss_before: Optional[float],
    gauss_after: Optional[float],
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Log and package before/after evaluation metrics."""

    summary: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    keys = ["psnr", "ssim", "lpips", "fps"]
    LOGGER.info("Real evaluation (before): gaussians=%s", gauss_before)
    LOGGER.info(
        "Before metrics: %s",
        {k: before.get(k) for k in keys},
    )
    LOGGER.info("Real evaluation (after): gaussians=%s", gauss_after)
    LOGGER.info(
        "After metrics: %s",
        {k: after.get(k) for k in keys},
    )
    for key in keys:
        summary[key] = (before.get(key), after.get(key))
    summary["gaussians"] = (gauss_before, gauss_after)
    return summary


def _plot_metrics(summary: Dict[str, Tuple[Optional[float], Optional[float]]], path: Path) -> None:
    """Plot before/after metrics if matplotlib is available."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        LOGGER.warning("matplotlib not available; skipping plot generation")
        return

    stages = ["before", "after"]
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for key, (before, after) in summary.items():
        if key == "gaussians":
            continue
        if before is None or after is None:
            continue
        ax.plot(stages, [before, after], marker="o", label=key.upper())
        plotted = True
    if not plotted:
        plt.close(fig)
        LOGGER.warning("Insufficient metrics for plotting; skipping")
        return
    ax.set_title("TTF Metrics Before/After")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    LOGGER.info("Saved metric plot to %s", path)


def _iter_optional_paths(values: Iterable[Optional[Path]]) -> Iterable[Path]:
    """Yield only valid paths from an iterable of optional paths."""

    for value in values:
        if isinstance(value, Path):
            yield value


def main() -> None:
    """Entry point for the CLI utility."""

    parser = argparse.ArgumentParser(description="Run TTF sanity checks")
    parser.add_argument("--metrics-before", type=Path, default=None, help="Metrics file prior to pruning")
    parser.add_argument("--metrics-after", type=Path, default=None, help="Metrics file after pruning")
    parser.add_argument("--gaussians-before", type=float, default=None, help="Manual gaussian count before")
    parser.add_argument("--gaussians-after", type=float, default=None, help="Manual gaussian count after")
    parser.add_argument("--plot-path", type=Path, default=Path("outputs/ttf_sanity/metrics.png"))
    parser.add_argument("--seed", type=int, default=0, help="Seed for the synthetic sanity test")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    LOGGER.info("Running synthetic prune_round sanity check")
    stats = run_random_sanity(seed=args.seed)
    LOGGER.info("Synthetic summary: %s", stats)

    before_metrics: Optional[Dict[str, float]] = None
    after_metrics: Optional[Dict[str, float]] = None
    for path in _iter_optional_paths([args.metrics_before, args.metrics_after]):
        if not path.exists():
            LOGGER.warning("Metrics file missing: %s", path)
    if args.metrics_before and args.metrics_before.exists():
        before_metrics = _parse_metrics(args.metrics_before)
    if args.metrics_after and args.metrics_after.exists():
        after_metrics = _parse_metrics(args.metrics_after)

    if before_metrics and after_metrics:
        gauss_before = _extract_gaussians(before_metrics, args.gaussians_before)
        gauss_after = _extract_gaussians(after_metrics, args.gaussians_after)
        summary = _summarise_real_metrics(before_metrics, after_metrics, gauss_before, gauss_after)
        _plot_metrics(summary, args.plot_path)
    else:
        LOGGER.info("Real metrics unavailable; skipping plot generation")


def _smoke_test() -> None:
    """Minimal smoke test executed when running this module directly."""

    logging.basicConfig(level=logging.INFO)
    stats = run_random_sanity(seed=42)
    assert stats["pruned"] <= stats["target"]
    LOGGER.info("Smoke test completed: %s", stats)


if __name__ == "__main__":
    main()
