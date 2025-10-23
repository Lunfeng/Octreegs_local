from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from octreegs.utils.ema import EmaTracker
from octreegs.utils.edge import pixels_to_gaussians, sobel_heat, topk_mask
from octreegs.utils.quantile import mixed_quantiles


class TtfController:
    """Controller implementing a lightweight version of the TTF pruning loop."""

    def __init__(
        self,
        model,
        trainer: Optional[object],
        cfg: Optional[Dict],
    ) -> None:
        self.model = model
        self.trainer = trainer
        self.cfg = cfg or {}
        common = self.cfg.get("common", {})
        self.lambda_dssim = common.get("lambda_dssim", 0.2)
        self.short_ft_iters = int(common.get("short_ft_iters", 800))
        self.final_ft_iters = int(common.get("final_ft_iters", 10000))
        self.rounds = int(common.get("rounds", 6))
        self.ema_momentum = float(common.get("ema_momentum", 0.9))
        self.edge_topk_percent = float(common.get("edge_topk_percent", 0.15))
        self.edge_boost = float(common.get("edge_boost", 1.35))
        self.leaf_q_reduce = float(common.get("leaf_q_reduce", 0.07))
        self.nmin_per_leaf = int(common.get("nmin_per_leaf", 18))
        self.rollback_enable = bool(common.get("rollback_enable", True))
        self.rollback_window_rounds = int(common.get("rollback_window_rounds", 2))
        self.rollback_ratio = float(common.get("rollback_ratio", 0.02))
        self.local_psnr_drop = float(common.get("local_psnr_drop", 0.3))
        self.weight_alpha = float(common.get("w_alpha", 1.0))
        self.weight_sigma = float(common.get("w_sigma", 0.5))
        self.weight_sh = float(common.get("w_sh", 0.25))

        self.grad_ema: Optional[EmaTracker] = None
        self.alpha_ema: Optional[EmaTracker] = None
        self.prune_mask: Optional[torch.BoolTensor] = None
        self.recently_pruned_round: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ helpers
    def _ensure_buffers(self) -> None:
        num = int(self.model.num_gaussians())
        device = self.model.get_anchor.device if hasattr(self.model, "get_anchor") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.grad_ema is None or self.grad_ema.num_items != num:
            self.grad_ema = EmaTracker(num, self.ema_momentum).to(device)
        else:
            self.grad_ema.resize_(num)
        if self.alpha_ema is None or self.alpha_ema.num_items != num:
            self.alpha_ema = EmaTracker(num, self.ema_momentum).to(device)
        else:
            self.alpha_ema.resize_(num)
        if self.prune_mask is None or self.prune_mask.shape[0] != num:
            self.prune_mask = torch.zeros(num, dtype=torch.bool, device=device)
        if self.recently_pruned_round is None or self.recently_pruned_round.shape[0] != num:
            self.recently_pruned_round = torch.full((num,), -1, dtype=torch.int32, device=device)

    def _gather_grad_norm(self, tensor: Optional[torch.Tensor]) -> torch.Tensor:
        if tensor is None:
            return torch.zeros(self.model.num_gaussians(), device=self.prune_mask.device)
        if tensor.grad is None:
            return torch.zeros(tensor.shape[0], device=tensor.device)
        grad = tensor.grad.detach()
        if grad.ndim == 1:
            return grad.abs()
        grad = grad.view(grad.shape[0], -1)
        return torch.norm(grad, dim=1)

    def _enforce_leaf_min(self, keep: torch.BoolTensor, leaf_ids: torch.LongTensor) -> torch.BoolTensor:
        if leaf_ids.numel() == 0:
            return keep
        nmin = max(self.nmin_per_leaf, 1)
        unique_ids = torch.unique(leaf_ids)
        for leaf in unique_ids:
            leaf_mask = leaf_ids == leaf
            num_keep = (keep & leaf_mask).sum()
            if num_keep >= nmin:
                continue
            deficit = nmin - num_keep
            indices = torch.nonzero(leaf_mask).flatten()
            if indices.numel() == 0:
                continue
            scores = self.alpha_ema.value[indices]
            order = torch.argsort(scores, descending=True)
            selected = indices[order[:deficit]]
            keep[selected] = True
        return keep

    def _build_edge_mask(self) -> Optional[torch.BoolTensor]:
        if self.trainer is None:
            return None
        eval_dict = self.trainer.evaluate(return_rgb=True, tiles=False)
        rgb = eval_dict.get("rgb_sampled")
        pix2gauss = eval_dict.get("pix2gauss_map")
        if rgb is None or pix2gauss is None:
            return None
        heat = sobel_heat(rgb)
        mask = topk_mask(heat, self.edge_topk_percent)
        gauss_mask = pixels_to_gaussians(mask, pix2gauss)
        if gauss_mask.numel() == 0:
            return None
        full_mask = torch.zeros_like(self.prune_mask)
        num = min(full_mask.shape[0], gauss_mask.shape[0])
        full_mask[:num] = gauss_mask[:num]
        return full_mask.bool()

    # ------------------------------------------------------------------ public API
    def stats_warmup(self, iters: int = 1000) -> None:
        del iters  # The current implementation uses instantaneous statistics.
        self._ensure_buffers()
        opacity = self.model.get_opacity().detach().reshape(-1)
        self.alpha_ema.update(opacity)
        grad_norm = torch.zeros_like(self.alpha_ema.value)
        self.grad_ema.update(grad_norm)
        self.model.set_prune_mask(self.prune_mask)

    def observe_after_backward(self) -> None:
        self._ensure_buffers()
        opacity = self.model.get_opacity().detach().reshape(-1)
        grad_alpha = self._gather_grad_norm(getattr(self.model, "_opacity", None))
        grad_sigma = self._gather_grad_norm(getattr(self.model, "_scaling", None))
        grad_sh = self._gather_grad_norm(getattr(self.model, "_anchor_feat", None))
        grad_total = (
            self.weight_alpha * grad_alpha[:opacity.shape[0]]
            + self.weight_sigma * grad_sigma[:opacity.shape[0]]
            + self.weight_sh * grad_sh[:opacity.shape[0]]
        )
        self.alpha_ema.update(opacity)
        self.grad_ema.update(grad_total)

    def prune_round(self, round_idx: int, alpha_q: float = 0.6, grad_q: float = 0.6, gamma_iter: Optional[float] = None) -> None:
        self._ensure_buffers()
        alive = ~self.prune_mask
        if alive.sum() == 0:
            return
        alpha_vals = self.alpha_ema.value
        grad_vals = self.grad_ema.value
        qa, qg = mixed_quantiles(alpha_vals[alive], grad_vals[alive], alpha_q, grad_q)
        g_eff = grad_vals.clone()
        edge_mask = self._build_edge_mask()
        if edge_mask is not None and edge_mask.shape[0] == g_eff.shape[0]:
            g_eff[edge_mask] *= self.edge_boost
        keep = (alpha_vals >= qa) | (g_eff >= qg)
        leaf_ids = self.model.leaf_ids
        if leaf_ids.shape[0] == keep.shape[0]:
            keep = self._enforce_leaf_min(keep, leaf_ids)
            high_var_leaf = self.model.leaf_high_variance_mask(leaf_ids)
            if high_var_leaf.numel() > 0:
                relaxed_qa = qa * (1.0 - self.leaf_q_reduce)
                relaxed_qg = qg * (1.0 - self.leaf_q_reduce)
                per_gauss_leaf = high_var_leaf[leaf_ids]
                relax_keep = (alpha_vals >= relaxed_qa) | (g_eff >= relaxed_qg)
                keep = keep | (per_gauss_leaf & relax_keep)
                keep = self._enforce_leaf_min(keep, leaf_ids)
        candidates = alive & (~keep)
        if candidates.sum() == 0:
            return
        gamma = float(gamma_iter if gamma_iter is not None else self.cfg.get("balanced", {}).get("gamma_iter", 0.325))
        target = max(int(math.ceil(alive.sum().item() * gamma)), 1)
        target = min(target, int(candidates.sum().item()))
        cand_indices = torch.nonzero(candidates).flatten()
        scores = 0.5 * alpha_vals[candidates].abs() + 0.5 * g_eff[candidates].abs()
        order = torch.argsort(scores)
        selected = cand_indices[order[:target]]
        self.prune_mask[selected] = True
        self.recently_pruned_round[selected] = round_idx
        self.model.set_prune_mask(self.prune_mask)

    def short_finetune(self, iters: Optional[int] = None) -> None:
        if self.trainer is None:
            return
        steps = int(iters or self.short_ft_iters)
        for _ in range(steps):
            self.trainer.step(use_mask=True, lambda_dssim=self.lambda_dssim)

    def final_finetune(self, iters: Optional[int] = None) -> None:
        if self.trainer is None:
            return
        steps = int(iters or self.final_ft_iters)
        for _ in range(steps):
            self.trainer.step(use_mask=False, lambda_dssim=self.lambda_dssim)

    def maybe_rollback(self, round_idx: int) -> None:
        if not self.rollback_enable or self.trainer is None:
            return
        eval_dict = self.trainer.evaluate(return_rgb=False, tiles=True)
        tile_delta = eval_dict.get("tile_psnr_delta")
        if tile_delta is None:
            return
        if (tile_delta < -self.local_psnr_drop).any():
            eligible = self.prune_mask & (self.recently_pruned_round >= round_idx - self.rollback_window_rounds)
            num_restore = int(self.rollback_ratio * eligible.sum().item())
            if num_restore <= 0:
                return
            cand = torch.nonzero(eligible).flatten()
            order = torch.argsort(self.recently_pruned_round[cand], descending=True)
            restore = cand[order[:num_restore]]
            self.prune_mask[restore] = False
            self.model.set_prune_mask(self.prune_mask)

    def run_online_phase(self, schedule: Dict[str, float]) -> None:
        rounds = int(schedule.get("rounds", 1))
        gamma_iter = schedule.get("gamma_iter")
        alpha_q = float(schedule.get("alpha_q", 0.6))
        grad_q = float(schedule.get("grad_q", 0.6))
        self._ensure_buffers()
        for r in range(rounds):
            self.prune_round(r, alpha_q=alpha_q, grad_q=grad_q, gamma_iter=gamma_iter)
            self.short_finetune(schedule.get("short_ft_iters", self.short_ft_iters))

    def run_posthoc_phase(self, preset: str = "balanced") -> None:
        self._ensure_buffers()
        preset_cfg = self.cfg.get(preset, {})
        gamma_iter = preset_cfg.get("gamma_iter", 0.325)
        alpha_q = preset_cfg.get("alpha_q", 0.6)
        grad_q = preset_cfg.get("grad_q", 0.6)
        rounds = preset_cfg.get("rounds", self.rounds)
        short_iters = preset_cfg.get("short_ft_iters", self.short_ft_iters)
        final_iters = preset_cfg.get("final_ft_iters", self.final_ft_iters)
        self.stats_warmup()
        for r in range(int(rounds)):
            self.prune_round(r, alpha_q=alpha_q, grad_q=grad_q, gamma_iter=gamma_iter)
            self.short_finetune(short_iters)
            self.maybe_rollback(r)
        self.model.finalize_prune(self.prune_mask)
        self._ensure_buffers()
        self.final_finetune(final_iters)

