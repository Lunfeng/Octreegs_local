"""Gradient-aware Trimming-the-Fat (TTF) pruning controller."""
from __future__ import annotations

import logging
import math
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

from octreegs.utils.ema import EmaTracker
from octreegs.utils.edge import pixels_to_gaussians, sobel_heat, topk_mask
from octreegs.utils.quantile import mixed_quantiles, robust_quantile


LOGGER = logging.getLogger(__name__)


class PruneRateScheduler:
    """Adaptive scheduler for the per-round pruning ratio ``γ_iter``."""

    def __init__(
        self,
        base: float = 0.325,
        minv: float = 0.18,
        maxv: float = 0.45,
        dpsnr_lo: float = -0.1,
        dpsnr_hi: float = -0.3,
    ) -> None:
        self.base = float(base)
        self.minv = float(minv)
        self.maxv = float(maxv)
        self.dpsnr_lo = float(dpsnr_lo)
        self.dpsnr_hi = float(dpsnr_hi)
        self.value = float(base)

    def step(self, dpsnr: float) -> float:
        """Return the next ``γ_iter`` based on global ``ΔPSNR``."""

        if math.isnan(dpsnr):
            return self.value
        drop = float(dpsnr)
        if drop <= self.dpsnr_hi:
            target = self.minv
        elif drop >= self.dpsnr_lo:
            target = self.maxv
        else:
            span = self.dpsnr_lo - self.dpsnr_hi
            ratio = 0.0 if span == 0 else (drop - self.dpsnr_hi) / span
            target = self.minv + ratio * (self.maxv - self.minv)
        target = max(self.minv, min(self.maxv, target))
        self.value = 0.7 * self.value + 0.3 * target
        self.value = max(self.minv, min(self.maxv, self.value))
        return self.value


class TtfController:
    """Coordinate online and post-hoc TTF pruning flows."""

    _alpha_priority: Optional[torch.Tensor] = None

    def __init__(
        self,
        model,
        renderer,
        train_step_fn: Callable,
        eval_step_fn: Callable,
        cfg: Dict,
        device: str | torch.device = "cuda",
    ) -> None:
        self.model = model
        self.renderer = renderer
        self.train_step_fn = train_step_fn
        self.eval_step_fn = eval_step_fn
        self.cfg = cfg
        param_device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
        self.device = torch.device(device) if device is not None else param_device

        self.common_cfg: Dict = dict(cfg.get("common", {}))
        ema_momentum = float(self.common_cfg.get("ema_momentum", 0.9))
        num_gaussians = int(model.num_gaussians())
        self.grad_ema = EmaTracker(num_gaussians, momentum=ema_momentum, device=self.device)
        self.alpha_ema = EmaTracker(num_gaussians, momentum=ema_momentum, device=self.device)
        self.prune_mask = torch.zeros(num_gaussians, dtype=torch.bool, device=self.device)
        self.recently_pruned_round = torch.zeros(num_gaussians, dtype=torch.int16, device=self.device)
        self.model.set_prune_mask(self.prune_mask)
        base_gamma = float(self.common_cfg.get("gamma_iter", 0.325))
        gamma_min = float(self.common_cfg.get("gamma_min", 0.18))
        gamma_max = float(self.common_cfg.get("gamma_max", base_gamma))
        dpsnr_lo = float(self.common_cfg.get("dpsnr_lo", -0.1))
        dpsnr_hi = float(self.common_cfg.get("dpsnr_hi", -0.3))
        self.scheduler = PruneRateScheduler(
            base=base_gamma,
            minv=gamma_min,
            maxv=max(gamma_min, gamma_max),
            dpsnr_lo=dpsnr_lo,
            dpsnr_hi=dpsnr_hi,
        )
        self.gamma_iter = base_gamma
        self.round_idx = 0
        self._stats_ready = False
        self._last_metrics: Dict[str, float] = {}
        self._baseline_metrics: Optional[Dict[str, float]] = None
        self._baseline_gaussians: int = num_gaussians
        self._pruned_history: list[Dict[str, torch.Tensor]] = []

    # ---------- helpers ----------
    @staticmethod
    def _robust_norm(x: torch.Tensor) -> torch.Tensor:
        """Return element-wise ``(x - median) / MAD`` (zeros when degenerate)."""

        if x.numel() == 0:
            return torch.zeros_like(x)
        median = x.median()
        mad = (x - median).abs().median()
        if not torch.isfinite(median) or mad <= 1e-8:
            return torch.zeros_like(x)
        return (x - median) / (mad + 1e-8)

    def _sh_energy(self) -> torch.Tensor:
        """Return texture strength ``[N]`` computed from SH coefficients."""

        sh = getattr(self.model, "gaussians", {}).get("sh")  # type: ignore[index]
        if sh is None:
            return torch.zeros_like(self.grad_ema.values())
        data = sh.detach()
        if data.dim() == 1:
            energy = data.abs()
        else:
            energy = data.view(data.shape[0], -1).pow(2).mean(dim=1)
        energy = energy.to(device=self.device, dtype=self.grad_ema.values().dtype)
        energy = energy - energy.min()
        denom = energy.max()
        if denom <= 1e-8:
            return torch.zeros_like(energy)
        return energy / (denom + 1e-8)

    def _ensure_buffers(self) -> None:
        """Resize controller buffers to match ``model.num_gaussians()``."""

        n = int(self.model.num_gaussians())
        if self.prune_mask.numel() == n:
            return
        ema_momentum = float(self.common_cfg.get("ema_momentum", 0.9))
        self.grad_ema = EmaTracker(n, momentum=ema_momentum, device=self.device)
        self.alpha_ema = EmaTracker(n, momentum=ema_momentum, device=self.device)
        self.prune_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.recently_pruned_round = torch.zeros(n, dtype=torch.int16, device=self.device)
        self.model.set_prune_mask(self.prune_mask)

    def _call_train(self, iters: int, **kwargs) -> Optional[Dict]:
        """Invoke the configured training step with compatibility fallbacks."""

        try:
            return self.train_step_fn(iters, **kwargs)
        except TypeError:
            # Fallback to legacy signature ``train_step_fn(use_mask, lambda_dssim)``
            use_mask = bool(kwargs.get("use_mask", True))
            lambda_dssim = kwargs.get("loss_weights", {}).get(
                "lambda_dssim", self.common_cfg.get("lambda_dssim", 0.2)
            )
            result = None
            for _ in range(int(iters)):
                result = self.train_step_fn(use_mask, lambda_dssim)
            return result

    def _call_eval(self, **kwargs) -> Dict:
        """Invoke evaluation with keyword compatibility fallbacks."""

        try:
            return self.eval_step_fn(**kwargs)
        except TypeError:
            if "return_rgb" in kwargs and "tiles" in kwargs:
                return self.eval_step_fn(kwargs["return_rgb"])  # type: ignore[misc]
            if "return_rgb" in kwargs:
                return self.eval_step_fn(kwargs["return_rgb"])  # type: ignore[misc]
            return self.eval_step_fn()  # type: ignore[misc]

    # ---------- statistics ----------
    @torch.no_grad()
    def stats_warmup(self, iters: int = 1000) -> None:
        """Bootstrap gradient/opacity EMAs via short optimisation."""

        lambda_dssim = float(self.common_cfg.get("lambda_dssim", 0.2))

        def _hook() -> None:
            with torch.no_grad():
                alpha_tensor = self.model.gaussians.get("alpha")  # type: ignore[index]
                if alpha_tensor is None:
                    raise KeyError("Model must expose 'alpha' tensor")
                alpha_flat = alpha_tensor.detach().view(alpha_tensor.shape[0], -1).mean(dim=1)
                alpha_norm = self._robust_norm(alpha_flat.to(self.device))
                g_alpha = self._collect_grad("alpha")
                g_cov = self._collect_grad(["cov", "scale", "R"])
                g_sh = self._collect_grad("sh")
                g_total = 1.0 * g_alpha + 0.5 * g_cov + 0.25 * g_sh
                self.grad_ema.update(g_total)
                self.alpha_ema.update(alpha_norm)

        self.model.freeze_growth(True)
        self._call_train(
            int(iters),
            loss_weights={"lambda_dssim": lambda_dssim},
            freeze_growth=True,
            use_mask=True,
            hook_after_backward=_hook,
        )
        self._clamp_alpha()
        self.model.freeze_growth(False)
        self._stats_ready = True

    def _collect_grad(self, keys) -> torch.Tensor:
        """Aggregate per-Gaussian gradient magnitudes for ``keys``."""

        if isinstance(keys, (list, tuple)):
            tensors = [self.model.gaussians.get(k) for k in keys if self.model.gaussians.get(k) is not None]
        else:
            tensors = [self.model.gaussians.get(keys)]  # type: ignore[list-item]
        buffer = torch.zeros_like(self.grad_ema.values())
        valid = False
        for tensor in tensors:
            if tensor is None or tensor.grad is None:
                continue
            grad = tensor.grad.detach().to(self.device)
            grad = grad.view(grad.shape[0], -1)
            buffer = buffer + grad.norm(p=2, dim=1)
            valid = True
            break
        return self._robust_norm(buffer) if valid else buffer

    # ---------- one round ----------
    @torch.no_grad()
    def prune_round(self, round_idx: int, alpha_q: float = 0.55, grad_q: float = 0.55) -> None:
        """Execute a single pruning round following the TTF heuristic stack."""

        self._ensure_buffers()
        if not self._stats_ready:
            self.stats_warmup(iters=int(self.common_cfg.get("stats_warmup_iters", 200)))

        self.round_idx = max(self.round_idx, int(round_idx))
        N = int(self.model.num_gaussians())
        leaf_ids = self.model.leaf_ids.to(self.device)
        alive = ~self.prune_mask

        render_pkg = self._render_edge_view()
        rgb = render_pkg.get("rgb")
        if rgb is None:
            mask_edge = torch.zeros(N, dtype=torch.bool, device=self.device)
        else:
            heat = sobel_heat(rgb.to(self.device))
            percent = float(self.common_cfg.get("edge_topk_percent", 0.15))
            high_mask = topk_mask(heat, percent)
            mask_edge = pixels_to_gaussians(
                high_mask,
                render_pkg.get("gauss_ids", torch.arange(N, device=self.device)),
                render_pkg.get("pix2gauss_ptr", torch.zeros(1, dtype=torch.long, device=self.device)),
                render_pkg.get("pix2gauss_idx", torch.zeros(0, dtype=torch.long, device=self.device)),
                render_pkg.get("pix_coords", torch.zeros(0, 2, dtype=torch.long, device=self.device)),
                N,
            )

        alpha_vals = self.alpha_ema.values().clone()
        grad_vals = self.grad_ema.values().clone()
        edge_boost = float(self.common_cfg.get("edge_boost", 1.6))
        grad_vals = grad_vals + 0.15 * self._sh_energy()
        grad_vals[mask_edge] *= edge_boost

        Qa, Qg = mixed_quantiles(alpha_vals[alive], grad_vals[alive], alpha_q, grad_q)
        keep = (alpha_vals >= Qa) | (grad_vals >= Qg)

        nmin_per_leaf = int(self.common_cfg.get("nmin_per_leaf", 24))
        TtfController._alpha_priority = alpha_vals
        keep = self._enforce_leaf_minimum(keep, nmin_per_leaf, leaf_ids)
        TtfController._alpha_priority = None

        high_var_leaf = self.model.leaf_high_variance_mask(leaf_ids)
        if high_var_leaf.numel() > 0:
            hv_mask = high_var_leaf[leaf_ids]
            if hv_mask.any():
                leaf_reduce = float(self.common_cfg.get("leaf_q_reduce", 0.10))
                Qa_offset = robust_quantile(alpha_vals[hv_mask], 0.5)
                Qg_offset = robust_quantile(grad_vals[hv_mask], 0.5)
                Qa_leaf = Qa - leaf_reduce * Qa_offset
                Qg_leaf = Qg - leaf_reduce * Qg_offset
                relaxed = (alpha_vals >= Qa_leaf) | (grad_vals >= Qg_leaf)
                keep = keep | (hv_mask & relaxed)

        score_ref = 0.5 * alpha_vals + 0.5 * grad_vals
        keep = self._enforce_leaf_budget(
            keep,
            leaf_ids,
            nmin=24,
            keep_ratio=0.30,
            score_ref=score_ref,
        )

        cand = alive & ~keep
        alive_count = int(alive.sum().item())
        num_target = min(int(cand.sum().item()), int(math.ceil(alive_count * float(self.gamma_iter))))
        chosen = torch.zeros_like(cand)
        if num_target > 0:
            cand_indices = torch.nonzero(cand, as_tuple=False).flatten()
            cand_scores = score_ref[cand_indices]
            if cand_scores.numel() <= num_target:
                selected = cand_indices
            else:
                order = torch.topk(cand_scores, num_target, largest=False).indices
                selected = cand_indices[order]
            chosen[selected] = True
            history_entry = {
                "round": torch.tensor(int(round_idx), dtype=torch.int16),
                "indices": selected.detach().cpu(),
                "scores": score_ref[selected].detach().cpu(),
                "alpha": alpha_vals[selected].detach().cpu(),
                "grad": grad_vals[selected].detach().cpu(),
            }
            self._pruned_history.append(history_entry)
            max_history = int(self.common_cfg.get("history_max_rounds", 12))
            if len(self._pruned_history) > max_history:
                self._pruned_history.pop(0)
            self.prune_mask[selected] = True
            self.recently_pruned_round[selected] = int(round_idx)
            alpha_buffer = self.alpha_ema.values()
            grad_buffer = self.grad_ema.values()
            alpha_buffer[selected] = 0.0
            grad_buffer[selected] = 0.0
        self.model.set_prune_mask(self.prune_mask)

        LOGGER.info(
            "TTF prune round %d: γ_iter=%.3f keep=%d cand=%d chosen=%d",
            round_idx,
            float(self.gamma_iter),
            int(keep.sum().item()),
            int(cand.sum().item()),
            int(chosen.sum().item()),
        )
        self.round_idx = int(round_idx)

    def _render_edge_view(self) -> Dict[str, torch.Tensor]:
        """Try obtaining a render package with pixel→Gaussian mapping."""

        if self.renderer is not None:
            view = self.cfg.get("edge_view")
            try:
                pkg = self.renderer.render_with_contrib(view)
                return self._to_device_pkg(pkg)
            except Exception:  # pragma: no cover - best effort fallback
                LOGGER.debug("render_with_contrib failed; falling back to eval_step_fn")
        eval_pkg = self._call_eval(return_rgb=True)
        pkg = {
            "rgb": eval_pkg.get("rgb") or eval_pkg.get("render"),
            "gauss_ids": eval_pkg.get("gauss_ids", torch.arange(self.model.num_gaussians(), device=self.device)),
            "pix2gauss_ptr": eval_pkg.get(
                "pix2gauss_ptr", torch.zeros(1, dtype=torch.long, device=self.device)
            ),
            "pix2gauss_idx": eval_pkg.get(
                "pix2gauss_idx", torch.zeros(0, dtype=torch.long, device=self.device)
            ),
            "pix_coords": eval_pkg.get(
                "pix_coords", torch.zeros(0, 2, dtype=torch.long, device=self.device)
            ),
        }
        return self._to_device_pkg(pkg)

    def _to_device_pkg(self, pkg: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move render package tensors to the controller device."""

        result: Dict[str, torch.Tensor] = dict(pkg)
        for key in ("gauss_ids", "pix2gauss_ptr", "pix2gauss_idx", "pix_coords"):
            tensor = result.get(key)
            if isinstance(tensor, torch.Tensor):
                result[key] = tensor.to(self.device)
        rgb = result.get("rgb")
        if isinstance(rgb, torch.Tensor):
            result["rgb"] = rgb.to(self.device)
        return result

    @staticmethod
    def _enforce_leaf_minimum(
        keep: torch.BoolTensor, nmin: int, leaf_ids: torch.LongTensor
    ) -> torch.BoolTensor:
        """Ensure at least ``nmin`` gaussians survive inside every leaf."""

        if leaf_ids.numel() == 0:
            return keep
        keep = keep.clone()
        scores = TtfController._alpha_priority
        for leaf in torch.unique(leaf_ids):
            mask = leaf_ids == leaf
            total = int(mask.sum().item())
            if total == 0:
                continue
            min_keep = min(int(nmin), total)
            current = int((keep & mask).sum().item())
            if current >= min_keep:
                continue
            need = min_keep - current
            candidates = torch.nonzero(mask & ~keep, as_tuple=False).flatten()
            if candidates.numel() == 0:
                keep[mask] = True
                continue
            if scores is not None and scores.numel() == keep.numel():
                order = torch.argsort(scores[candidates], descending=True)
                chosen = candidates[order[:need]]
            else:
                chosen = candidates[:need]
            keep[chosen] = True
        return keep

    @staticmethod
    def _enforce_leaf_budget(
        keep: torch.BoolTensor,
        leaf_ids: torch.LongTensor,
        nmin: int,
        keep_ratio: float,
        score_ref: torch.Tensor,
    ) -> torch.BoolTensor:
        """Ensure per-leaf keep count respects minimums and score priorities."""

        if leaf_ids.numel() == 0:
            return keep
        keep = keep.clone()
        for leaf in torch.unique(leaf_ids):
            mask = leaf_ids == leaf
            total = int(mask.sum().item())
            if total == 0:
                continue
            required = max(int(nmin), int(math.ceil(total * keep_ratio)))
            if required >= total:
                keep[mask] = True
                continue
            indices = torch.nonzero(mask, as_tuple=False).flatten()
            scores = score_ref[indices]
            order = torch.argsort(scores, descending=True)
            selected = indices[order[:required]]
            keep[selected] = True
        return keep

    # ---------- rollback ----------
    @torch.no_grad()
    def maybe_rollback(self, round_idx: int, tile_psnr_drop: Optional[torch.Tensor]) -> None:
        """Restore a fraction of recently pruned gaussians if quality tanks locally."""

        if not self.common_cfg.get("rollback_enable", True):
            return
        if tile_psnr_drop is None:
            return
        drops = tile_psnr_drop.to(self.device)
        threshold = float(self.common_cfg.get("local_psnr_drop", 0.2))
        if not torch.is_floating_point(drops) or not (drops < -threshold).any():
            return
        window = int(self.common_cfg.get("rollback_window_rounds", 2))
        ratio = float(self.common_cfg.get("rollback_ratio", 0.05))
        restored = self._restore_recent(window, ratio)
        if restored > 0:
            LOGGER.info("TTF rollback restored %d gaussians", restored)

    def _restore_recent(self, max_age: int, ratio: float) -> int:
        """Restore a fraction of recently pruned gaussians."""

        if ratio <= 0.0:
            return 0
        if not self._pruned_history:
            return 0
        restored = 0
        alpha_buffer = self.alpha_ema.values()
        grad_buffer = self.grad_ema.values()
        for entry in reversed(self._pruned_history):
            round_tensor = entry.get("round")
            entry_round = int(round_tensor.item()) if isinstance(round_tensor, torch.Tensor) else int(round_tensor)
            if self.round_idx - entry_round > max_age:
                continue
            indices = entry.get("indices")
            if indices is None:
                continue
            idx = torch.as_tensor(indices, device=self.device, dtype=torch.long)
            if idx.numel() == 0:
                continue
            scores = torch.as_tensor(entry.get("scores"), device=self.device, dtype=alpha_buffer.dtype)
            alpha_vals = torch.as_tensor(entry.get("alpha"), device=self.device, dtype=alpha_buffer.dtype)
            grad_vals = torch.as_tensor(entry.get("grad"), device=self.device, dtype=grad_buffer.dtype)
            num_restore = max(int(math.ceil(idx.numel() * ratio)), 1)
            order = torch.argsort(scores, descending=True)
            take = order[:num_restore]
            chosen = idx.take(take)
            if chosen.numel() == 0:
                continue
            self.prune_mask[chosen] = False
            self.recently_pruned_round[chosen] = 0
            alpha_buffer[chosen] = alpha_vals.take(take)
            grad_buffer[chosen] = grad_vals.take(take)
            restored += int(chosen.numel())
            keep_mask = torch.ones(idx.shape[0], dtype=torch.bool, device=self.device)
            keep_mask[take] = False
            entry["indices"] = idx[keep_mask].cpu()
            entry["scores"] = scores[keep_mask].cpu()
            entry["alpha"] = alpha_vals[keep_mask].cpu()
            entry["grad"] = grad_vals[keep_mask].cpu()
            break
        if restored > 0:
            self.model.set_prune_mask(self.prune_mask)
        return restored

    # ---------- schedule ----------
    def short_finetune(self, iters: int) -> None:
        """Run a short masked fine-tuning stage."""

        if iters <= 0:
            return
        lambda_dssim = float(self.common_cfg.get("lambda_dssim", 0.2))
        self._call_train(
            int(iters),
            loss_weights={"lambda_dssim": lambda_dssim},
            freeze_growth=True,
            use_mask=True,
            hook_after_backward=None,
        )
        self._clamp_alpha()

    def final_finetune(self, iters: int) -> None:
        """Run the final fine-tuning stage after hard pruning."""

        if iters <= 0:
            return
        lambda_dssim = float(self.common_cfg.get("lambda_dssim", 0.2))
        self._call_train(
            int(iters),
            loss_weights={"lambda_dssim": lambda_dssim},
            freeze_growth=True,
            use_mask=True,
            hook_after_backward=None,
        )
        self._clamp_alpha()

    def run_online_phase(
        self,
        rounds_per_call: int = 1,
        gamma_iter: float = 0.275,
        alpha_q: float = 0.55,
        grad_q: float = 0.55,
        short_ft_iters: int = 600,
    ) -> None:
        """Execute lightweight pruning during training (growth window callback)."""

        self.model.freeze_growth(True)
        previous_gamma = self.gamma_iter
        self.gamma_iter = float(gamma_iter)
        for _ in range(int(rounds_per_call)):
            self.prune_round(self.round_idx + 1, alpha_q=alpha_q, grad_q=grad_q)
            self.short_finetune(short_ft_iters)
        self.model.freeze_growth(False)
        self.gamma_iter = previous_gamma

    def run_posthoc_phase(self, preset: str = "balanced") -> Dict[str, float]:
        """Execute the full offline TTF pruning pipeline and return final metrics."""

        warmup_iters = int(self.common_cfg.get("stats_warmup_iters", 800))
        self.stats_warmup(warmup_iters)
        preset_cfg = dict(self.cfg.get(preset, {}))
        self.gamma_iter = float(preset_cfg.get("gamma_iter", self.scheduler.value))
        rounds = int(self.common_cfg.get("rounds", 6))
        short_iters = int(self.common_cfg.get("short_ft_iters", 800))
        alpha_q = float(preset_cfg.get("alpha_q", self.common_cfg.get("alpha_q", 0.55)))
        grad_q = float(preset_cfg.get("grad_q", self.common_cfg.get("grad_q", 0.55)))
        baseline_raw = self._call_eval(return_rgb=False, tiles=True)
        self._baseline_metrics = self._metrics_to_float_dict(baseline_raw)
        self._baseline_gaussians = int(self.model.num_gaussians())
        prev_psnr = self._baseline_metrics.get("psnr")
        LOGGER.info(
            "TTF baseline metrics: PSNR=%.3f SSIM=%.3f LPIPS=%.3f", 
            float(self._baseline_metrics.get("psnr", 0.0)),
            float(self._baseline_metrics.get("ssim", 0.0)),
            float(self._baseline_metrics.get("lpips", 0.0)),
        )
        for i in range(rounds):
            self.prune_round(self.round_idx + 1, alpha_q=alpha_q, grad_q=grad_q)
            self.short_finetune(short_iters)
            metrics = self._call_eval(return_rgb=False, tiles=True)
            psnr = float(metrics.get("psnr", 0.0))
            dpsnr = 0.0 if prev_psnr is None else psnr - prev_psnr
            tile_delta = metrics.get("tile_psnr_delta")
            triggered = False
            if isinstance(tile_delta, torch.Tensor):
                self.maybe_rollback(self.round_idx, tile_delta)
                if (tile_delta < -0.3).any():
                    new_gamma = max(float(self.gamma_iter) * 0.9, 0.18)
                    if new_gamma < self.gamma_iter - 1e-6:
                        LOGGER.warning(
                            "TTF round %d triggered rollback; reducing γ_iter from %.3f to %.3f",
                            i + 1,
                            float(self.gamma_iter),
                            new_gamma,
                        )
                    self.gamma_iter = new_gamma
                    self.scheduler.value = self.gamma_iter
                    triggered = True
            quality_trigger = self._quality_guard(metrics)
            if quality_trigger:
                metrics = self._call_eval(return_rgb=False, tiles=True)
                psnr = float(metrics.get("psnr", 0.0))
                dpsnr = 0.0 if prev_psnr is None else psnr - prev_psnr
                tile_delta = metrics.get("tile_psnr_delta")
                if isinstance(tile_delta, torch.Tensor):
                    self.maybe_rollback(self.round_idx, tile_delta)
                triggered = True
            if not triggered:
                self.gamma_iter = self.scheduler.step(dpsnr)
            prev_psnr = psnr
            LOGGER.info(
                "TTF round %d/%d metrics: PSNR=%.3f SSIM=%.3f LPIPS=%.3f FPS=%.2f",
                i + 1,
                rounds,
                float(metrics.get("psnr", 0.0)),
                float(metrics.get("ssim", 0.0)),
                float(metrics.get("lpips", 0.0)),
                float(metrics.get("fps", 0.0)),
            )
            cleaned: Dict[str, float] = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    cleaned[k] = float(v)
                elif isinstance(v, torch.Tensor) and v.numel() == 1:
                    cleaned[k] = float(v.item())
            self._last_metrics = cleaned

        self.model.finalize_prune(self.prune_mask)
        self._ensure_buffers()
        self._pruned_history.clear()
        final_iters = int(self.common_cfg.get("final_ft_iters", 10_000))
        self.final_finetune(final_iters)
        final_metrics = self._call_eval(return_rgb=False, tiles=True)
        summary: Dict[str, float] = {}
        for key, value in final_metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    summary[key] = float(value.item())
            elif isinstance(value, (int, float)):
                summary[key] = float(value)
        return summary

    def _clamp_alpha(self) -> None:
        """Clamp gaussian alpha tensor to avoid vanishing opacities."""

        alpha_tensor = getattr(self.model, "gaussians", {}).get("alpha")  # type: ignore[index]
        if isinstance(alpha_tensor, torch.Tensor):
            alpha_tensor.data.clamp_(min=1e-4)

    def _quality_guard(self, metrics: Dict[str, torch.Tensor | float | None]) -> bool:
        """Ensure quality deltas stay within tolerance and trigger restoration if needed."""

        if not self._baseline_metrics:
            return False
        psnr_thresh = float(self.common_cfg.get("guard_psnr_drop", 0.2))
        ssim_thresh = float(self.common_cfg.get("guard_ssim_drop", 0.003))
        lpips_thresh = float(self.common_cfg.get("guard_lpips_increase", 0.01))

        psnr = metrics.get("psnr")
        ssim = metrics.get("ssim")
        lpips = metrics.get("lpips")
        triggered = False

        if isinstance(psnr, torch.Tensor):
            psnr_val = float(psnr.detach().item()) if psnr.numel() == 1 else None
        else:
            psnr_val = float(psnr) if isinstance(psnr, (int, float)) else None
        if psnr_val is not None:
            baseline_psnr = self._baseline_metrics.get("psnr")
            if baseline_psnr is not None and psnr_val - baseline_psnr < -psnr_thresh:
                triggered = True

        if isinstance(ssim, torch.Tensor):
            ssim_val = float(ssim.detach().item()) if ssim.numel() == 1 else None
        else:
            ssim_val = float(ssim) if isinstance(ssim, (int, float)) else None
        if ssim_val is not None:
            baseline_ssim = self._baseline_metrics.get("ssim")
            if baseline_ssim is not None and ssim_val - baseline_ssim < -ssim_thresh:
                triggered = True

        if isinstance(lpips, torch.Tensor):
            lpips_val = float(lpips.detach().item()) if lpips.numel() == 1 else None
        else:
            lpips_val = float(lpips) if isinstance(lpips, (int, float)) else None
        if lpips_val is not None:
            baseline_lpips = self._baseline_metrics.get("lpips")
            if baseline_lpips is not None and lpips_val - baseline_lpips > lpips_thresh:
                triggered = True

        if not triggered:
            return False

        restore_ratio = float(self.common_cfg.get("guard_restore_ratio", 0.12))
        restore_window = int(self.common_cfg.get("guard_restore_rounds", 2))
        restored = self._restore_recent(restore_window, restore_ratio)
        if restored > 0:
            LOGGER.warning(
                "TTF quality guard restored %d gaussians (baseline PSNR=%.3f)",
                restored,
                float(self._baseline_metrics.get("psnr", 0.0)),
            )
        else:
            LOGGER.warning("TTF quality guard triggered but no gaussians restored")
        new_gamma = max(self.gamma_iter * 0.8, self.scheduler.minv)
        if new_gamma < self.gamma_iter - 1e-6:
            LOGGER.warning(
                "TTF quality guard reducing γ_iter from %.3f to %.3f", self.gamma_iter, new_gamma
            )
        self.gamma_iter = new_gamma
        self.scheduler.value = self.gamma_iter
        return True

    @staticmethod
    def _metrics_to_float_dict(metrics: Dict[str, object]) -> Dict[str, float]:
        """Convert metric dictionary entries into scalars when possible."""

        result: Dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    result[key] = float(value.item())
            elif isinstance(value, (int, float)):
                result[key] = float(value)
        return result


def _smoke_test() -> None:
    """Minimal smoke test for ``TtfController`` pruning logic."""

    class _DummyModel(torch.nn.Module):
        def __init__(self, N: int) -> None:
            super().__init__()
            self.gaussians = {
                "alpha": torch.rand(N, 1),
                "scale": torch.rand(N, 3),
                "sh": torch.rand(N, 16),
            }
            self.register_buffer("_leaf_ids", torch.randint(0, 32, (N,), dtype=torch.long))
            self.register_buffer("_mask", torch.zeros(N, dtype=torch.bool))

        def num_gaussians(self) -> int:
            return int(self._leaf_ids.numel())

        @property
        def leaf_ids(self) -> torch.Tensor:
            return self._leaf_ids

        def set_prune_mask(self, mask: torch.BoolTensor) -> None:
            self._mask = mask.clone()

        def finalize_prune(self, mask: torch.BoolTensor) -> None:
            keep = ~mask
            for key in list(self.gaussians.keys()):
                self.gaussians[key] = self.gaussians[key][keep]
            self._leaf_ids = self._leaf_ids[keep]
            self._mask = torch.zeros_like(self._leaf_ids, dtype=torch.bool)

        def freeze_growth(self, flag: bool) -> None:
            pass

        def leaf_high_variance_mask(self, leaf_ids: torch.Tensor) -> torch.Tensor:
            if leaf_ids.numel() == 0:
                return torch.zeros(0, dtype=torch.bool)
            num_leaves = int(leaf_ids.max().item()) + 1
            mask = torch.zeros(num_leaves, dtype=torch.bool)
            if num_leaves > 0:
                mask[: num_leaves // 8 + 1] = True
            return mask

    class _DummyRenderer:
        def __init__(self, N: int) -> None:
            self.N = N

        def render_with_contrib(self, view) -> Dict[str, torch.Tensor]:
            H = W = 32
            rgb = torch.rand(H, W, 3)
            gauss_ids = torch.arange(self.N, dtype=torch.long)
            M = 128
            pix_coords = torch.randint(0, H, (M, 2), dtype=torch.long)
            ptr = torch.arange(0, M + 1, dtype=torch.long)
            idx = torch.randint(0, self.N, (M,), dtype=torch.long)
            return {
                "rgb": rgb,
                "gauss_ids": gauss_ids,
                "pix2gauss_ptr": ptr,
                "pix2gauss_idx": idx,
                "pix_coords": pix_coords,
            }

    def _train_step(iters, **kwargs):
        hook = kwargs.get("hook_after_backward")
        if hook is not None:
            hook()
        return {"iters": iters}

    def _eval_step(return_rgb: bool = False, tiles: bool = False):
        data = {"psnr": 30.0, "ssim": 0.9, "lpips": 0.05, "fps": 24.0}
        if return_rgb:
            data.update(
                {
                    "rgb": torch.rand(8, 8, 3),
                    "gauss_ids": torch.arange(N),
                    "pix2gauss_ptr": torch.tensor([0, 2, 4], dtype=torch.long),
                    "pix2gauss_idx": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                    "pix_coords": torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
                }
            )
        if tiles:
            data["tile_psnr_delta"] = torch.tensor([-0.05, -0.4])
        return data

    N = 1000
    model = _DummyModel(N)
    renderer = _DummyRenderer(N)
    cfg = {
        "common": {
            "lambda_dssim": 0.2,
            "ema_momentum": 0.5,
            "edge_topk_percent": 0.2,
            "edge_boost": 1.2,
            "leaf_q_reduce": 0.05,
            "nmin_per_leaf": 10,
            "rounds": 2,
            "rollback_enable": True,
            "rollback_ratio": 0.5,
            "rollback_window_rounds": 2,
            "local_psnr_drop": 0.2,
            "short_ft_iters": 10,
            "final_ft_iters": 10,
        },
        "balanced": {"gamma_iter": 0.3},
    }
    controller = TtfController(model, renderer, _train_step, _eval_step, cfg, device="cpu")
    controller.grad_ema.update(torch.rand(N))
    controller.alpha_ema.update(torch.rand(N))
    controller.prune_round(1)
    mask_before = controller.prune_mask.clone()
    controller.maybe_rollback(1, torch.tensor([-0.5, -0.05]))
    mask_after = controller.prune_mask
    assert mask_after.sum() <= mask_before.sum()
    LOGGER.info("Smoke test keep=%d pruned=%d", int((~mask_after).sum()), int(mask_after.sum()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _smoke_test()
