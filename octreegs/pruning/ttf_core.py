"""Trimming-the-Fat (TTF) controller integration for OctreeGS."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

from octreegs.models.octree_model import OctreeGSModel
from octreegs.utils.ema import EmaTracker
from octreegs.utils.edge import pixels_to_gaussians, sobel_heat, topk_mask
from octreegs.utils.quantile import mixed_quantiles

TrainStepFn = Callable[[bool, float], Dict]
EvalStepFn = Callable[[bool, bool], Dict]


@dataclass
class TtfConfig:
    lambda_dssim: float = 0.2
    short_ft_iters: int = 800
    final_ft_iters: int = 10_000
    rounds: int = 6
    ema_momentum: float = 0.9
    edge_topk_percent: float = 0.15
    edge_boost: float = 1.35
    leaf_q_reduce: float = 0.07
    nmin_per_leaf: int = 18
    rollback_enable: bool = True
    rollback_window_rounds: int = 2
    rollback_ratio: float = 0.02
    local_psnr_drop: float = 0.3
    gamma_iter: float = 0.325


class TtfController:
    """Coordinate TTF pruning rounds for online/offline phases."""

    def __init__(
        self,
        model: OctreeGSModel,
        train_step_fn: TrainStepFn,
        eval_step_fn: EvalStepFn,
        cfg: TtfConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.train_step_fn = train_step_fn
        self.eval_step_fn = eval_step_fn
        self.cfg = cfg
        try:
            param_device = next(model.parameters()).device
        except StopIteration:
            param_device = torch.device("cpu")
        self.device = device or param_device

        num_gaussians = model.num_gaussians()
        self.grad_ema = EmaTracker(num_gaussians, momentum=cfg.ema_momentum, device=self.device)
        self.alpha_ema = EmaTracker(num_gaussians, momentum=cfg.ema_momentum, device=self.device)
        self.prune_mask = model.prune_mask.clone().to(self.device)
        self.recently_pruned_round = torch.zeros_like(self.prune_mask, dtype=torch.long)
        self.round_idx = 0

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def hook_after_backward(self) -> None:
        """Collect gradient and opacity statistics after a backward pass."""

        self._ensure_tracker_size()
        alpha = self.model.gaussians.get("alpha")
        if alpha is None:
            raise KeyError("Model must expose an 'alpha' tensor for TTF pruning")
        alpha_now = alpha.detach()
        if alpha_now.dim() > 1:
            alpha_now = alpha_now.view(alpha_now.shape[0], -1).mean(dim=1)
        grads = []
        for key in ("alpha", "cov", "sh"):
            tensor = self.model.gaussians.get(key)
            grad = None if tensor is None else tensor.grad
            if grad is None:
                grads.append(torch.zeros_like(alpha_now))
                continue
            g = grad.view(grad.shape[0], -1).norm(p=2, dim=1)
            grads.append(g)
        g_alpha, g_cov, g_sh = grads
        g_total = 1.0 * g_alpha + 0.5 * g_cov + 0.25 * g_sh
        self.grad_ema.update(g_total.detach())
        self.alpha_ema.update(alpha_now.detach())

    # ------------------------------------------------------------------
    # Core routines
    # ------------------------------------------------------------------
    def stats_warmup(self, iters: int = 1000) -> None:
        self._ensure_tracker_size()
        for _ in range(iters):
            self.train_step_fn(use_mask=False, lambda_dssim=self.cfg.lambda_dssim)
            self.hook_after_backward()

    def prune_round(self, round_idx: Optional[int] = None, alpha_q: float = 0.6, grad_q: float = 0.6,
                    gamma_iter: Optional[float] = None) -> Dict[str, float]:
        self._ensure_tracker_size()
        self.round_idx = round_idx if round_idx is not None else (self.round_idx + 1)
        alive = self.model.alive_mask()
        num_alive = int(alive.sum().item())
        if num_alive == 0:
            return {"num_pruned": 0, "num_alive": 0}

        Qa, Qg = mixed_quantiles(self.alpha_ema.value[alive], self.grad_ema.value[alive], alpha_q, grad_q)
        g_eff = self.grad_ema.value.clone()
        edge_mask = self._edge_sensitive_mask()
        if edge_mask.numel() == g_eff.numel():
            g_eff[edge_mask] *= self.cfg.edge_boost

        high_var_leaf = self.model.leaf_high_variance_mask(self.model.leaf_ids)
        keep = torch.zeros_like(alive)
        if alive.any():
            keep = (self.alpha_ema.value >= Qa) | (g_eff >= Qg)
            keep = self._enforce_leaf_minimum(keep, self.cfg.nmin_per_leaf)
            if high_var_leaf.numel() > 0:
                Qa_l = Qa * (1.0 - self.cfg.leaf_q_reduce)
                Qg_l = Qg * (1.0 - self.cfg.leaf_q_reduce)
                leaf_relax = (self.alpha_ema.value >= Qa_l) | (g_eff >= Qg_l)
                leaf_mask = high_var_leaf[self.model.leaf_ids]
                keep = keep | (leaf_mask & leaf_relax)
                keep = self._enforce_leaf_minimum(keep, self.cfg.nmin_per_leaf)

        cand = alive & (~keep)
        gamma = gamma_iter if gamma_iter is not None else self.cfg.gamma_iter
        num_target = max(int(math.ceil(num_alive * gamma)), 1)
        cand_idx = cand.nonzero().flatten()
        if cand_idx.numel() == 0:
            return {"num_pruned": 0, "num_alive": num_alive}
        score = 0.5 * self.alpha_ema.value[cand] + 0.5 * g_eff[cand]
        order = torch.argsort(score)
        chosen = cand_idx[order[:num_target]]
        self.prune_mask[chosen] = True
        self.recently_pruned_round[chosen] = self.round_idx
        self.model.set_prune_mask(self.prune_mask)
        return {
            "num_pruned": int(chosen.numel()),
            "num_alive": int(self.model.alive_mask().sum().item()),
        }

    def short_finetune(self, iters: Optional[int] = None) -> None:
        steps = iters if iters is not None else self.cfg.short_ft_iters
        for _ in range(steps):
            self.train_step_fn(use_mask=True, lambda_dssim=self.cfg.lambda_dssim)
            self.hook_after_backward()

    def final_finetune(self, iters: Optional[int] = None) -> None:
        steps = iters if iters is not None else self.cfg.final_ft_iters
        for _ in range(steps):
            self.train_step_fn(use_mask=False, lambda_dssim=self.cfg.lambda_dssim)
            self.hook_after_backward()

    def maybe_rollback(self, tile_psnr_delta: Optional[torch.Tensor] = None) -> int:
        if not self.cfg.rollback_enable:
            return 0
        if tile_psnr_delta is None:
            metrics = self.eval_step_fn(return_rgb=False, tiles=True)
            tile_psnr_delta = metrics.get("tile_psnr_delta")
        if tile_psnr_delta is None or tile_psnr_delta.numel() == 0:
            return 0
        if (tile_psnr_delta < -self.cfg.local_psnr_drop).any():
            eligible = self.prune_mask & (
                self.recently_pruned_round >= self.round_idx - self.cfg.rollback_window_rounds
            )
            num_restore = int(math.ceil(eligible.sum().item() * self.cfg.rollback_ratio))
            if num_restore <= 0:
                return 0
            cand = eligible.nonzero().flatten()
            _, order = torch.sort(self.recently_pruned_round[cand], descending=True)
            restore = cand[order[:num_restore]]
            self.prune_mask[restore] = False
            self.model.set_prune_mask(self.prune_mask)
            return int(restore.numel())
        return 0

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------
    def run_online_phase(self, schedule: Dict) -> None:
        rounds = int(schedule.get("rounds", 1))
        gamma = schedule.get("gamma_iter")
        alpha_q = float(schedule.get("alpha_q", 0.6))
        grad_q = float(schedule.get("grad_q", 0.6))
        short_iters = int(schedule.get("short_ft_iters", self.cfg.short_ft_iters))

        self.model.freeze_growth(True)
        for r in range(rounds):
            stats = self.prune_round(round_idx=r, alpha_q=alpha_q, grad_q=grad_q, gamma_iter=gamma)
            self.short_finetune(short_iters)
            metrics = self.eval_step_fn(return_rgb=False, tiles=False)
            print(f"[TTF][Online] Round {r}: pruned {stats['num_pruned']} -> alive {stats['num_alive']}")
            print(f"[TTF][Online] Metrics: {metrics}")
        self.model.freeze_growth(False)

    def run_posthoc_phase(self, preset: str) -> None:
        print(f"[TTF][Posthoc] Starting warmup ({self.cfg.rounds} rounds preset={preset})")
        self.stats_warmup(1000)
        for r in range(self.cfg.rounds):
            stats = self.prune_round(round_idx=r, alpha_q=0.6, grad_q=0.6)
            self.short_finetune(self.cfg.short_ft_iters)
            metrics = self.eval_step_fn(return_rgb=True, tiles=True)
            restored = self.maybe_rollback(metrics.get("tile_psnr_delta"))
            print(
                f"[TTF][Posthoc] Round {r}: pruned {stats['num_pruned']} -> alive {stats['num_alive']} (restored {restored})"
            )
            self._log_round_metrics(r, metrics)
        self.model.finalize_prune(self.prune_mask)
        self._ensure_tracker_size()
        self.final_finetune(self.cfg.final_ft_iters)
        final_metrics = self.eval_step_fn(return_rgb=True, tiles=True)
        self._log_final_metrics(preset, final_metrics)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _log_round_metrics(self, round_idx: int, metrics: Dict) -> None:
        psnr = self._safe_float(metrics.get("psnr"))
        ssim = self._safe_float(metrics.get("ssim"))
        lpips = self._safe_float(metrics.get("lpips"))
        fps = self._safe_float(metrics.get("fps"))
        print(
            f"[TTF] Round {round_idx} metrics: PSNR={psnr:.3f} SSIM={ssim:.3f} LPIPS={lpips:.3f} FPS={fps:.2f}"
        )

    def _log_final_metrics(self, preset: str, metrics: Dict) -> None:
        path = f"final_metrics_{preset}.txt"
        with open(path, "w", encoding="utf-8") as f:
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.mean().item()
                f.write(f"{key}: {value}\n")
        print(f"[TTF] Final metrics saved to {path}")

    def _safe_float(self, value: Optional[float]) -> float:
        if value is None:
            return float("nan")
        if isinstance(value, torch.Tensor):
            return float(value.mean().item())
        return float(value)

    def _edge_sensitive_mask(self) -> torch.Tensor:
        metrics = self.eval_step_fn(return_rgb=True, tiles=False)
        rgb = metrics.get("rgb_sampled")
        aux = metrics.get("pix2gauss_map")
        if rgb is None:
            return torch.zeros_like(self.prune_mask)
        if not isinstance(rgb, torch.Tensor):
            rgb = torch.as_tensor(rgb, dtype=torch.float32)
        heat = sobel_heat(rgb)
        topk = topk_mask(heat, self.cfg.edge_topk_percent)
        gauss_mask = pixels_to_gaussians(topk, aux)
        if gauss_mask.numel() == 0:
            return torch.zeros_like(self.prune_mask)
        if gauss_mask.numel() != self.prune_mask.numel():
            padded = torch.zeros_like(self.prune_mask)
            limit = min(padded.numel(), gauss_mask.numel())
            padded[:limit] = gauss_mask[:limit]
            return padded.bool()
        return gauss_mask.bool()

    def _enforce_leaf_minimum(self, keep_mask: torch.Tensor, nmin: int) -> torch.Tensor:
        leaf_ids = self.model.leaf_ids
        result = keep_mask.clone()
        if leaf_ids.numel() == 0:
            return result
        num_leaves = int(leaf_ids.max().item()) + 1
        for leaf in range(num_leaves):
            idx = (leaf_ids == leaf).nonzero().flatten()
            if idx.numel() == 0:
                continue
            alive_mask = ~self.prune_mask[idx]
            keep_alive = result[idx] & alive_mask
            if keep_alive.sum() >= nmin:
                continue
            deficit = int(nmin - keep_alive.sum().item())
            candidates = alive_mask & (~result[idx])
            if candidates.sum() == 0:
                continue
            scores = self.alpha_ema.value[idx]
            order = torch.argsort(scores, descending=True)
            selected = []
            for pos in order.tolist():
                if candidates[pos]:
                    selected.append(idx[pos])
                if len(selected) >= deficit:
                    break
            if selected:
                selected_idx = torch.tensor(selected, device=result.device, dtype=torch.long)
                result[selected_idx] = True
        return result

    def _ensure_tracker_size(self) -> None:
        current = self.model.num_gaussians()
        if current == self.grad_ema.num_items:
            return
        self.grad_ema = EmaTracker(current, momentum=self.cfg.ema_momentum, device=self.device)
        self.alpha_ema = EmaTracker(current, momentum=self.cfg.ema_momentum, device=self.device)
        self.prune_mask = torch.zeros(current, dtype=torch.bool, device=self.device)
        self.recently_pruned_round = torch.zeros(current, dtype=torch.long, device=self.device)
        self.model.set_prune_mask(self.prune_mask)
