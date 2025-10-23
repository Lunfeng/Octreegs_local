"""Training engine with optional TTF pruning integration."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import torch
from torch.optim import Adam

from octreegs.pruning.ttf_core import TtfController


LOGGER = logging.getLogger(__name__)


class Trainer:
    """Wrapper coordinating training and TTF pruning.

    The trainer expects external callables for the actual rendering/optimisation
    logic. This keeps the module agnostic of the specifics of the OctreeGS
    implementation while providing the hooks required by the pruning controller.
    """

    def __init__(
        self,
        model,
        optimizer,
        renderer,
        cfg: Optional[Any] = None,
        ttf_options: Optional[Any] = None,
        train_step_impl: Optional[Callable[[bool, float], Dict]] = None,
        eval_step_impl: Optional[Callable[[bool, bool], Dict]] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or {}
        self.optimizer = self._build_optimizer() if optimizer is None else optimizer
        if optimizer is not None:
            LOGGER.debug("Using caller supplied optimizer instance")
        if ttf_options is not None:
            LOGGER.debug("Trainer received legacy ttf_options argument; prefer cfg['ttf'] settings")
        self.renderer = renderer
        self._train_step_impl = train_step_impl
        self._eval_step_impl = eval_step_impl
        self.ttf: Optional[TtfController] = None
        self._prev_eval_psnr: Optional[float] = None
        self._prev_tile_psnr: Optional[torch.Tensor] = None
        self._initialise_ttf_controller()

    # ------------------------------------------------------------------
    # Public API used by the pruning controller
    # ------------------------------------------------------------------
    def step(self, use_mask: bool, lambda_dssim: float) -> Dict:
        """Execute a single optimisation step.

        Parameters
        ----------
        use_mask:
            Whether the model's prune mask should be respected during rendering.
        lambda_dssim:
            Weight for the DSSIM term used during short fine-tuning.
        """

        if self._train_step_impl is None:
            raise NotImplementedError("Trainer.step requires a train_step_impl callable")
        return self._train_step_impl(use_mask, lambda_dssim)

    def _ttf_train_step(self, iters: int, **kwargs) -> Optional[Dict]:
        """Adapter exposing the trainer's step function to the TTF controller."""

        if self._train_step_impl is None:
            raise NotImplementedError("Trainer.step requires a train_step_impl callable")
        use_mask = bool(kwargs.get("use_mask", True))
        lambda_dssim = kwargs.get("loss_weights", {}).get("lambda_dssim", 0.2)
        hook = kwargs.get("hook_after_backward")
        result: Optional[Dict] = None
        for _ in range(int(iters)):
            result = self._train_step_impl(use_mask, lambda_dssim)
            if callable(hook):
                hook()
        return result

    def evaluate(self, return_rgb: bool = False, tiles: bool = False) -> Dict:
        if self._eval_step_impl is None:
            raise NotImplementedError("Trainer.evaluate requires an eval_step_impl callable")
        raw = self._eval_step_impl(return_rgb, tiles)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise TypeError("Evaluation callable must return a dictionary")

        result: Dict[str, Any] = {}

        def _as_scalar(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    return float(value.item())
                return value.detach().clone()
            if isinstance(value, (int, float)):
                return float(value)
            return value

        # Required metrics always included in the response.
        for key in ("psnr", "ssim", "lpips", "fps"):
            if key in raw:
                result[key] = _as_scalar(raw[key])
            else:
                result[key] = None

        if tiles:
            tile_scores = raw.get("tile_psnr")
            if isinstance(tile_scores, torch.Tensor):
                tile_scores = tile_scores.detach()
                if self._prev_tile_psnr is not None and self._prev_tile_psnr.shape == tile_scores.shape:
                    delta = tile_scores - self._prev_tile_psnr
                else:
                    delta = torch.zeros_like(tile_scores)
                self._prev_tile_psnr = tile_scores.clone()
                result["tile_psnr_delta"] = delta
            else:
                result["tile_psnr_delta"] = _as_scalar(raw.get("tile_psnr_delta"))

            current_psnr = result.get("psnr")
            if isinstance(current_psnr, (int, float)):
                previous = self._prev_eval_psnr
                delta_psnr = 0.0 if previous is None else float(current_psnr) - previous
                result["delta_psnr"] = delta_psnr
                self._prev_eval_psnr = float(current_psnr)
            else:
                result["delta_psnr"] = _as_scalar(raw.get("delta_psnr"))

        if return_rgb:
            rgb_tensor = raw.get("rgb_sampled") or raw.get("rgb")
            if rgb_tensor is not None:
                result["rgb_sampled"] = rgb_tensor
            keys = ("gauss_ids", "pix2gauss_ptr", "pix2gauss_idx", "pix_coords")
            if all(k in raw for k in keys):
                for key in keys:
                    result[key] = raw[key]
            else:
                view = raw.get("view")
                if view is not None and hasattr(self.renderer, "render_with_contrib"):
                    contrib = self.renderer.render_with_contrib(view)
                    result.setdefault("rgb_sampled", contrib.get("rgb"))
                    for key in keys:
                        if key in contrib:
                            result[key] = contrib[key]
                else:
                    for key in keys:
                        result.setdefault(key, None)

        # Preserve any additional keys from the evaluation output.
        for key, value in raw.items():
            if key not in result:
                result[key] = value
        return result

    # ------------------------------------------------------------------
    # Integration helpers
    # ------------------------------------------------------------------
    def on_growth_window_end(self) -> None:
        if self.ttf is None:
            return
        ttf_cfg = self._cfg_section("ttf")
        if not ttf_cfg.get("enable_online", True):
            return
        gamma_iter = float(ttf_cfg.get("gamma_iter", 0.275))
        alpha_q = float(ttf_cfg.get("alpha_q", 0.55))
        grad_q = float(ttf_cfg.get("grad_q", 0.55))
        short_iters = int(ttf_cfg.get("short_ft_iters", 600))
        self.model.freeze_growth(True)
        try:
            LOGGER.info("TTF online round start (gamma=%.3f)", gamma_iter)
            self.ttf.run_online_phase(
                rounds_per_call=1,
                gamma_iter=gamma_iter,
                alpha_q=alpha_q,
                grad_q=grad_q,
                short_ft_iters=short_iters,
            )
        finally:
            self.model.freeze_growth(False)

    def run_posthoc(self) -> None:
        if self.ttf is None:
            return
        ttf_cfg = self._cfg_section("ttf")
        if not ttf_cfg.get("enable_posthoc", True):
            return
        preset = ttf_cfg.get("preset", "balanced")
        self.model.freeze_growth(True)
        try:
            metrics = self.ttf.run_posthoc_phase(preset)
            LOGGER.info("Post-hoc TTF completed with metrics: %s", metrics)
        finally:
            self.model.freeze_growth(False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _initialise_ttf_controller(self) -> None:
        if self.renderer is None:
            return
        section = self._cfg_section("ttf")
        controller_cfg = self._extract_controller_cfg(section)
        if controller_cfg:
            self.ttf = TtfController(self.model, self.renderer, self._ttf_train_step, self.evaluate, controller_cfg)

    def _cfg_section(self, key: str) -> Dict[str, Any]:
        if isinstance(self.cfg, dict):
            return dict(self.cfg.get(key, {}))
        if hasattr(self.cfg, "get"):
            try:
                value = self.cfg.get(key, {})  # type: ignore[attr-defined]
                if isinstance(value, dict):
                    return dict(value)
            except Exception:  # pragma: no cover - defensive
                pass
        if hasattr(self.cfg, key):
            value = getattr(self.cfg, key)
            if isinstance(value, dict):
                return dict(value)
        return {}

    def _extract_controller_cfg(self, section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        keys_to_ignore = {"enable_online", "enable_posthoc", "preset", "gamma_iter", "alpha_q", "grad_q", "short_ft_iters"}
        for candidate in ("controller_cfg", "config", "ttf_config"):
            cfg_obj = section.get(candidate)
            if isinstance(cfg_obj, dict):
                return cfg_obj
        filtered = {k: v for k, v in section.items() if k not in keys_to_ignore}
        if "common" in filtered:
            return filtered
        return None

    def _build_optimizer(self) -> Adam:
        """Create an Adam optimizer with parameter-specific learning rates."""

        low_lr_params = []
        high_lr_params = []
        other_params = []
        if hasattr(self.model, "gaussians"):
            for name, param in self.model.gaussians.items():  # type: ignore[attr-defined]
                if not isinstance(param, torch.nn.Parameter) or not param.requires_grad:
                    continue
                lname = name.lower()
                if any(key in lname for key in ("pos", "position", "scale", "rotation", "rot")):
                    low_lr_params.append(param)
                elif any(key in lname for key in ("alpha", "opacity", "sh")):
                    high_lr_params.append(param)
                else:
                    other_params.append(param)

        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            if (
                param not in low_lr_params
                and param not in high_lr_params
                and param not in other_params
            ):
                other_params.append(param)

        param_groups = []
        if low_lr_params:
            param_groups.append({"params": low_lr_params, "lr": 1e-3})
        if high_lr_params:
            param_groups.append({"params": high_lr_params, "lr": 3e-3})
        if other_params:
            param_groups.append({"params": other_params, "lr": 3e-3})

        if not param_groups:
            raise RuntimeError("No trainable parameters found for optimizer")

        return Adam(param_groups, betas=(0.9, 0.99))
