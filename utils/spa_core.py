import csv
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import Tensor


class SpaManager:
    def __init__(
        self,
        device: torch.device,
        logger,
        *,
        spa_start_iter: int,
        spa_stop_iter: int,
        spa_delta_start: float = 3e-4,
        spa_delta_end: float = 1e-3,
        spa_interval_warm: int = 60,
        spa_interval_stable: int = 40,
        quota_update_interval: int = 1000,
        kappa_total: Optional[int] = None,
        keep_ratio: Optional[float] = None,
        alpha_occupancy: float = 0.6,
        anchor_m_min: int = 1,
        age_grace_iters: int = 2000,
        hysteresis_M_out: int = 3,
        hysteresis_M_in: int = 2,
        new_level_bootstrap_ratio: float = 0.1,
        hot_anchor_boost: float = 1.0,
        log_dir: Optional[str] = None,
        exp_name: Optional[str] = None,
    ) -> None:
        self.device = device
        self.logger = logger
        self.spa_start_iter = int(spa_start_iter)
        self.spa_stop_iter = int(spa_stop_iter)
        self.spa_delta_start = float(spa_delta_start)
        self.spa_delta_end = float(spa_delta_end)
        self.spa_interval_warm = int(spa_interval_warm)
        self.spa_interval_stable = int(spa_interval_stable)
        self.quota_update_interval = int(quota_update_interval)
        self.kappa_total = kappa_total
        self.keep_ratio = keep_ratio
        self.alpha_occupancy = float(alpha_occupancy)
        self.anchor_m_min = int(anchor_m_min)
        self.age_grace_iters = int(age_grace_iters)
        self.hysteresis_M_out = int(hysteresis_M_out)
        self.hysteresis_M_in = int(hysteresis_M_in)
        self.new_level_bootstrap_ratio = float(new_level_bootstrap_ratio)
        self.hot_anchor_boost = float(hot_anchor_boost)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.exp_name = exp_name

        self.z: Optional[Tensor] = None
        self.lam: Optional[Tensor] = None
        self.age: Optional[Tensor] = None
        self.vis_hit_ema: Optional[Tensor] = None
        self.grad_ema: Optional[Tensor] = None
        self.level: Optional[Tensor] = None
        self.anchor_id: Optional[Tensor] = None
        self.alive_counter: Optional[Tensor] = None
        self.dead_counter: Optional[Tensor] = None

        self.N: int = 0
        self.level_first_seen: Dict[int, int] = {}
        self.level_quota: Dict[int, int] = {}
        self._last_quota_iter: Optional[int] = None
        self._last_iteration: Optional[int] = None
        self._csv_path: Optional[Path] = None
        self._json_path: Optional[Path] = None
        self._csv_header_written: bool = False
        self._csv_fields: List[str] = [
            "iteration",
            "delta",
            "interval",
            "num_gaussians",
            "kappa_total",
            "keep_ratio",
            "kappa_descriptor",
            "anchor_m_min",
            "age_grace_iters",
            "diff_norm",
            "anchor_empty_rate",
            "kappa_per_level",
            "selected_ratio_per_level",
            "selected_counts_per_level",
            "candidate_counts_per_level",
            "anchor_empty_rate_per_level",
            "thresholds_per_level",
        ]

        self.enabled = True
        self._fallback_single_layer = False
        self._fallback_warned = False

    def start(
        self,
        a0: Tensor,
        level: Optional[Tensor] = None,
        anchor_id: Optional[Tensor] = None,
    ) -> None:
        a0 = a0.to(self.device)
        self.N = int(a0.shape[0])
        self.z = a0.detach().clone()
        self.lam = torch.zeros_like(self.z, device=self.device)
        self.age = torch.zeros(self.N, device=self.device, dtype=torch.float32)
        self.vis_hit_ema = torch.zeros(self.N, device=self.device, dtype=torch.float32)
        self.grad_ema = torch.zeros(self.N, device=self.device, dtype=torch.float32)
        self.alive_counter = torch.zeros(self.N, device=self.device, dtype=torch.long)
        self.dead_counter = torch.zeros(self.N, device=self.device, dtype=torch.long)
        missing_level = level is None
        missing_anchor = anchor_id is None
        self._fallback_single_layer = missing_level or missing_anchor

        if self._fallback_single_layer:
            self.level = torch.zeros(self.N, device=self.device, dtype=torch.long)
            self.anchor_id = torch.zeros(self.N, device=self.device, dtype=torch.long)
            if not self._fallback_warned:
                self._log(
                    "[SPA][WARN] level/anchor_id not provided, fallback to single-layer top-k."
                )
                self._fallback_warned = True
        else:
            self.level = level.to(self.device).long()
            self.anchor_id = anchor_id.to(self.device).long()
        self.level_first_seen.clear()
        self.level_quota.clear()
        self._last_quota_iter = None
        self._last_iteration = None
        self._register_level_first_seen(self.spa_start_iter)
        self._prepare_logging_files()
        self._log_start()

    def append_loss(self, loss: Tensor, a: Tensor, iteration: int) -> Tensor:
        self._ensure_started()
        delta_val = self.delta(iteration)
        residual = a - self.z + self.lam
        penalty = 0.5 * delta_val * torch.sum(residual * residual)
        return loss + penalty

    def ingest_visibility(self, visible_mask: Tensor) -> None:
        self._ensure_started()
        visible_mask = visible_mask.to(self.device, dtype=torch.float32)
        self.vis_hit_ema.mul_(0.95).add_(0.05 * visible_mask)

    def ingest_grad(self, point_grad_norm: Tensor) -> None:
        self._ensure_started()
        point_grad_norm = point_grad_norm.to(self.device, dtype=torch.float32)
        self.grad_ema.mul_(0.9).add_(0.1 * point_grad_norm)

    def set_meta(
        self,
        level: Optional[Tensor],
        anchor_id: Optional[Tensor],
    ) -> None:
        if level is not None:
            self.level = level.to(self.device).long()
            self._register_level_first_seen(self._last_iteration or self.spa_start_iter)
        if anchor_id is not None:
            self.anchor_id = anchor_id.to(self.device).long()

    def step_prox(self, a: Tensor, iteration: int) -> None:
        self._ensure_started()
        current_interval = self.interval(iteration)
        increment = 0 if self._last_iteration is None else max(1, iteration - self._last_iteration)
        self.age += float(increment)
        self._last_iteration = iteration
        a = a.to(self.device)
        combined = (a + self.lam).abs().squeeze(-1)
        if combined.dim() == 0:
            combined = combined.unsqueeze(0)

        levels = self._get_levels()
        anchors = self._get_anchors()
        unique_levels = torch.unique(levels)
        level_totals = self._candidate_counts_per_level(levels, unique_levels)

        if self._fallback_single_layer:
            quota_total = min(self._current_kappa_total(), self.N)
            self.level_quota.clear()
            if self.N > 0:
                self.level_quota[0] = int(quota_total)
            self._last_quota_iter = iteration
        elif self._should_update_quota(iteration):
            self._update_quota(unique_levels, iteration)

        selected = torch.zeros(self.N, device=self.device, dtype=torch.bool)
        selected_counts: Dict[int, int] = {int(l.item()): 0 for l in unique_levels}

        for level_value in unique_levels:
            level_mask = levels == level_value
            if not self._fallback_single_layer:
                anchor_ids_level = anchors[level_mask]
                anchor_unique = torch.unique(anchor_ids_level)

                # Anchor floor selection
                for anchor_value in anchor_unique:
                    anchor_mask = (anchors == anchor_value) & level_mask
                    anchor_indices = torch.nonzero(anchor_mask, as_tuple=False).squeeze(-1)
                    if anchor_indices.numel() == 0:
                        continue
                    quota_anchor = min(self.anchor_m_min, anchor_indices.numel())
                    if quota_anchor > 0:
                        anchor_scores = combined[anchor_indices]
                        topk = torch.topk(anchor_scores, quota_anchor, sorted=False).indices
                        selected_indices = anchor_indices[topk]
                        selected[selected_indices] = True
            # Age grace
            young_mask = level_mask & (self.age < float(self.age_grace_iters))
            selected[young_mask] = True

            level_quota = self.level_quota.get(int(level_value.item()), self.anchor_m_min)
            current_selected = int((selected & level_mask).sum().item())
            remaining_quota = max(level_quota - current_selected, 0)
            if remaining_quota <= 0:
                selected_counts[int(level_value.item())] = current_selected
                continue

            candidate_mask = (~selected) & level_mask
            if not torch.any(candidate_mask):
                selected_counts[int(level_value.item())] = current_selected
                continue

            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
            scores = self._compute_scores(
                candidate_indices,
                combined,
                anchors,
            )
            if candidate_indices.numel() <= remaining_quota:
                selected[candidate_indices] = True
            else:
                topk_indices = torch.topk(scores, remaining_quota, sorted=False).indices
                selected_indices = candidate_indices[topk_indices]
                selected[selected_indices] = True
            selected_counts[int(level_value.item())] = int((selected & level_mask).sum().item())

        # Two-stage projection result
        new_z = torch.zeros_like(self.z)
        new_z[selected] = a[selected]
        self.z = new_z

        # Update counters with hysteresis
        alive_mask = selected
        dead_mask = ~selected
        self.alive_counter[alive_mask] += 1
        self.alive_counter[dead_mask] = 0
        self.dead_counter[dead_mask] += 1
        self.dead_counter[alive_mask] = 0

        # Multiplier update
        self.lam = self.lam + (a - self.z)

        # Logging
        delta_val = self.delta(iteration)
        num_gaussians = self.N
        kappa_total = self._current_kappa_total()
        level_stats = self._selected_ratio_per_level(selected_counts, level_totals)
        diff_norm = torch.norm(a - self.z).item()
        kappa_map = self._kappa_per_level(unique_levels)
        anchor_empty_rate = self._anchor_empty_rate_per_level(anchors, selected, levels, unique_levels)
        kappa_descriptor = self._kappa_descriptor(kappa_total)
        thresholds = self._quantiles_per_level(combined, levels, selected, unique_levels)
        self._log_step(
            iteration,
            delta_val,
            current_interval,
            num_gaussians,
            kappa_descriptor,
            kappa_map,
            level_stats,
            diff_norm,
            anchor_empty_rate,
            thresholds,
        )
        self._persist_step_metrics(
            iteration,
            delta_val,
            current_interval,
            num_gaussians,
            kappa_total,
            level_stats,
            diff_norm,
            anchor_empty_rate,
            kappa_descriptor,
            kappa_map,
            selected_counts,
            level_totals,
            thresholds,
        )

    def build_prune_mask(self) -> Tensor:
        self._ensure_started()
        return (self.z.squeeze(-1) == 0) & (self.dead_counter >= self.hysteresis_M_out)

    def apply_prune_mask(self, prune_mask: Tensor) -> None:
        self._ensure_started()
        prune_mask = prune_mask.to(self.device, dtype=torch.bool)
        if prune_mask.numel() != self.N:
            raise ValueError("Prune mask size mismatch with managed tensor count")
        keep_mask = ~prune_mask
        keep_total = int(keep_mask.sum().item())
        if keep_total == self.N:
            return

        def _mask_optional(tensor: Optional[Tensor]) -> Optional[Tensor]:
            if tensor is None:
                return None
            return tensor[keep_mask]

        self.z = self.z[keep_mask]
        self.lam = self.lam[keep_mask]
        self.age = self.age[keep_mask]
        self.vis_hit_ema = self.vis_hit_ema[keep_mask]
        self.grad_ema = self.grad_ema[keep_mask]
        self.alive_counter = self.alive_counter[keep_mask]
        self.dead_counter = self.dead_counter[keep_mask]
        self.level = _mask_optional(self.level)
        self.anchor_id = _mask_optional(self.anchor_id)
        self.N = keep_total

        remaining_levels = set()
        if self.N > 0:
            for lvl in torch.unique(self._get_levels()).tolist():
                remaining_levels.add(int(lvl))
        for lvl in list(self.level_first_seen.keys()):
            if lvl not in remaining_levels:
                self.level_first_seen.pop(lvl, None)
        for lvl in list(self.level_quota.keys()):
            if lvl not in remaining_levels:
                self.level_quota.pop(lvl, None)

    def remaining_per_level(self) -> Dict[int, int]:
        self._ensure_started()
        if self.N == 0:
            return {}
        levels = self._get_levels()
        if levels.numel() == 0:
            return {0: 0}
        unique_levels, counts = torch.unique(levels, return_counts=True)
        return {int(lvl.item()): int(cnt.item()) for lvl, cnt in zip(unique_levels, counts)}

    def snapshot_logs(self, tag: str = "pruned") -> List[Path]:
        if self.log_dir is None:
            return []
        snapshots: List[Path] = []
        for path in (self._csv_path, self._json_path):
            if path is None or not path.exists():
                continue
            snapshot_path = path.with_name(f"{path.stem}_{tag}{path.suffix}")
            shutil.copy2(path, snapshot_path)
            snapshots.append(snapshot_path)
        return snapshots

    def resize_on_density(self, newN: int, a: Tensor) -> None:
        self._ensure_started()
        if newN <= self.N:
            return
        a = a.to(self.device)
        addN = newN - self.N
        pad_shape = (addN,) + self.z.shape[1:]
        self.z = torch.cat([self.z, a[self.N:newN].detach().clone()], dim=0)
        self.lam = torch.cat([self.lam, torch.zeros(pad_shape, device=self.device, dtype=self.lam.dtype)], dim=0)
        self.age = torch.cat([self.age, torch.zeros(addN, device=self.device)], dim=0)
        self.vis_hit_ema = torch.cat([self.vis_hit_ema, torch.zeros(addN, device=self.device)], dim=0)
        self.grad_ema = torch.cat([self.grad_ema, torch.zeros(addN, device=self.device)], dim=0)
        self.alive_counter = torch.cat([
            self.alive_counter,
            torch.zeros(addN, device=self.device, dtype=torch.long),
        ], dim=0)
        self.dead_counter = torch.cat([
            self.dead_counter,
            torch.zeros(addN, device=self.device, dtype=torch.long),
        ], dim=0)
        if self.level is not None:
            last_level = self.level[-1]
            level_pad = torch.full((addN,), int(last_level.item()), device=self.device, dtype=torch.long)
            self.level = torch.cat([self.level, level_pad], dim=0)
        if self.anchor_id is not None:
            last_anchor = self.anchor_id[-1]
            anchor_pad = torch.full((addN,), int(last_anchor.item()), device=self.device, dtype=torch.long)
            self.anchor_id = torch.cat([self.anchor_id, anchor_pad], dim=0)
        self.N = newN

    def interval(self, iteration: int) -> int:
        if iteration < self.spa_start_iter + 3000:
            return self.spa_interval_warm
        return self.spa_interval_stable

    # Helper methods
    def delta(self, iteration: int) -> float:
        if iteration <= self.spa_start_iter:
            return self.spa_delta_start
        if iteration >= self.spa_stop_iter:
            return self.spa_delta_end
        span = self.spa_stop_iter - self.spa_start_iter
        if span <= 0:
            return self.spa_delta_end
        ratio = (iteration - self.spa_start_iter) / span
        return self.spa_delta_start + ratio * (self.spa_delta_end - self.spa_delta_start)

    def _ensure_started(self) -> None:
        if self.z is None or self.lam is None:
            raise RuntimeError("SpaManager.start must be called before using the manager")

    def _get_levels(self) -> Tensor:
        if self.level is not None:
            return self.level
        return torch.zeros(self.N, device=self.device, dtype=torch.long)

    def _get_anchors(self) -> Tensor:
        if self.anchor_id is not None:
            return self.anchor_id
        levels = self._get_levels()
        return levels

    def _should_update_quota(self, iteration: int) -> bool:
        if self._last_quota_iter is None:
            return True
        return (iteration - self._last_quota_iter) >= self.quota_update_interval

    def _update_quota(self, unique_levels: Tensor, iteration: int) -> None:
        kappa_total = self._current_kappa_total()
        if kappa_total <= 0:
            kappa_total = self.N
        occ_map: Dict[int, float] = {}
        err_map: Dict[int, float] = {}
        anchors = self._get_anchors()
        levels = self._get_levels()

        for level_value in unique_levels:
            level_mask = levels == level_value
            occ = float(self.vis_hit_ema[level_mask].sum().item())
            err = float(self.grad_ema[level_mask].sum().item())
            occ_map[int(level_value.item())] = occ
            err_map[int(level_value.item())] = err

        occ_total = sum(occ_map.values())
        err_total = sum(err_map.values())
        for lvl in unique_levels:
            key = int(lvl.item())
            occ_norm = occ_map[key] / occ_total if occ_total > 0 else 1.0 / max(len(unique_levels), 1)
            err_norm = err_map[key] / err_total if err_total > 0 else 1.0 / max(len(unique_levels), 1)
            weight = self.alpha_occupancy * occ_norm + (1.0 - self.alpha_occupancy) * err_norm
            level_quota = weight * kappa_total
            min_quota = self._min_quota_for_level(key, anchors, levels)
            bootstrap_quota = self._bootstrap_quota(key, iteration, kappa_total)
            level_quota = max(level_quota, min_quota, bootstrap_quota)
            max_allowed = int((levels == lvl).sum().item())
            self.level_quota[key] = min(int(math.ceil(level_quota)), max_allowed)

        self._last_quota_iter = iteration

    def _min_quota_for_level(self, level_value: int, anchors: Tensor, levels: Tensor) -> int:
        level_mask = levels == level_value
        anchor_mask = anchors[level_mask]
        if anchor_mask.numel() == 0:
            return 0
        unique_anchors = torch.unique(anchor_mask)
        num_active = unique_anchors.numel()
        return int(num_active * self.anchor_m_min)

    def _bootstrap_quota(self, level_value: int, iteration: int, kappa_total: int) -> int:
        first_seen = self.level_first_seen.get(level_value, iteration)
        if iteration - first_seen <= 2000:
            return int(math.ceil(self.new_level_bootstrap_ratio * kappa_total))
        return 0

    def _compute_scores(
        self,
        candidate_indices: Tensor,
        combined: Tensor,
        anchors: Tensor,
    ) -> Tensor:
        amplitude = combined[candidate_indices]
        vis = self.vis_hit_ema[candidate_indices]
        grad = self.grad_ema[candidate_indices]
        amp_norm = amplitude / (amplitude.max().clamp(min=1e-6))
        vis_norm = vis / (vis.max().clamp(min=1e-6))
        grad_norm = grad / (grad.max().clamp(min=1e-6))
        score = amp_norm * vis_norm * (1.0 + grad_norm)
        if self.hot_anchor_boost != 1.0:
            anchor_subset = anchors[candidate_indices]
            hot_mask = self._hot_anchor_mask(anchor_subset)
            boost = torch.ones_like(score)
            boost[hot_mask] = self.hot_anchor_boost
            score = score * boost
        return score

    def _hot_anchor_mask(self, anchor_subset: Tensor) -> Tensor:
        anchors = self._get_anchors()
        unique_anchors = torch.unique(anchors)
        if unique_anchors.numel() == 0:
            return torch.zeros_like(anchor_subset, dtype=torch.bool)
        anchor_vis = []
        for anchor_value in unique_anchors:
            mask = anchors == anchor_value
            anchor_vis.append(float(self.vis_hit_ema[mask].mean().item()))
        global_mean = sum(anchor_vis) / max(len(anchor_vis), 1)
        hot_map: Dict[int, bool] = {}
        for anchor_value, vis_mean in zip(unique_anchors.tolist(), anchor_vis):
            hot_map[int(anchor_value)] = vis_mean > global_mean
        hot_mask = torch.zeros_like(anchor_subset, dtype=torch.bool)
        for idx, anchor_value in enumerate(anchor_subset.tolist()):
            hot_mask[idx] = hot_map.get(int(anchor_value), False)
        return hot_mask.to(self.device)

    def _current_kappa_total(self) -> int:
        if self.kappa_total is not None:
            return int(self.kappa_total)
        if self.keep_ratio is not None:
            return int(math.ceil(self.keep_ratio * self.N))
        return self.N

    def current_kappa_total(self) -> int:
        return self._current_kappa_total()

    def _selected_ratio_per_level(
        self,
        selected_counts: Dict[int, int],
        level_totals: Dict[int, int],
    ) -> Dict[int, Dict[str, float]]:
        result: Dict[int, Dict[str, float]] = {}
        if not level_totals and self.N > 0:
            level_totals = {0: self.N}
        for lvl, total in level_totals.items():
            denom = max(int(total), 1)
            selected = int(selected_counts.get(lvl, 0))
            ratio = float(selected) / float(denom)
            result[int(lvl)] = {
                "selected": float(selected),
                "total": float(denom),
                "ratio": ratio,
            }
        return result

    def _candidate_counts_per_level(
        self,
        levels: Tensor,
        unique_levels: Tensor,
    ) -> Dict[int, int]:
        if unique_levels.numel() == 0:
            return {}
        counts = {}
        for lvl in unique_levels:
            mask = levels == lvl
            counts[int(lvl.item())] = int(mask.sum().item())
        return counts

    def _register_level_first_seen(self, iteration: int) -> None:
        if self.level is None:
            self.level_first_seen[0] = iteration
            return
        for lvl in torch.unique(self.level).tolist():
            if lvl not in self.level_first_seen:
                self.level_first_seen[int(lvl)] = iteration

    def _log_start(self) -> None:
        kappa_descriptor = self._kappa_descriptor(self._current_kappa_total())
        message = (
            "[SPA] start | N=%d | delta_start=%.4g | delta_end=%.4g | interval_warm=%d | interval_stable=%d | %s"
            % (
                self.N,
                self.spa_delta_start,
                self.spa_delta_end,
                self.spa_interval_warm,
                self.spa_interval_stable,
                kappa_descriptor,
            )
        )
        self._log(message)

    def _log_step(
        self,
        iteration: int,
        delta_val: float,
        interval_val: int,
        num_gaussians: int,
        kappa_descriptor: str,
        kappa_map: Dict[int, int],
        level_stats: Dict[int, Dict[str, float]],
        diff_norm: float,
        anchor_empty_rate: Dict[int, float],
        thresholds: Dict[int, Dict[str, float]],
    ) -> None:
        ratio_str = ", ".join(
            f"L{lvl}: {int(stat['selected'])}/{int(stat['total'])} ({stat['ratio']:.2f})"
            for lvl, stat in sorted(level_stats.items())
        )
        kappa_str = ", ".join(
            f"L{lvl}: {quota}" for lvl, quota in sorted(kappa_map.items())
        )
        anchor_str = ", ".join(
            f"L{lvl}: {rate:.3f}" for lvl, rate in sorted(anchor_empty_rate.items())
        )
        threshold_str = ", ".join(
            "L{lvl}: p50={p50:.4g}, p90={p90:.4g}, p99={p99:.4g}".format(
                lvl=lvl,
                p50=vals.get("p50", 0.0),
                p90=vals.get("p90", 0.0),
                p99=vals.get("p99", 0.0),
            )
            for lvl, vals in sorted(thresholds.items())
        )
        message = (
            "[SPA] iter=%d | delta=%.4g | interval=%d | #G=%d | %s | kappa={%s} | "
            "anchor_m_min=%d | age_grace=%d | diff=%.4g | ratios={%s} | anchor_empty={%s} | thresholds={%s}"
            % (
                iteration,
                delta_val,
                interval_val,
                num_gaussians,
                kappa_descriptor,
                kappa_str,
                self.anchor_m_min,
                self.age_grace_iters,
                diff_norm,
                ratio_str,
                anchor_str,
                threshold_str,
            )
        )
        self._log(message)

    def _log(self, message: str) -> None:
        if self.logger is not None and hasattr(self.logger, "info"):
            self.logger.info(message)
        else:
            print(message)

    def _prepare_logging_files(self) -> None:
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.log_dir / "spa_metrics.csv"
        self._json_path = self.log_dir / "spa_metrics.jsonl"
        if self._csv_path.exists():
            self._csv_path.unlink()
        if self._json_path.exists():
            self._json_path.unlink()
        self._csv_header_written = False

    def _persist_step_metrics(
        self,
        iteration: int,
        delta_val: float,
        interval_val: int,
        num_gaussians: int,
        kappa_total: int,
        level_stats: Dict[int, Dict[str, float]],
        diff_norm: float,
        anchor_empty_rate: Dict[int, float],
        kappa_descriptor: str,
        kappa_map: Dict[int, int],
        selected_counts: Dict[int, int],
        level_totals: Dict[int, int],
        thresholds: Dict[int, Dict[str, float]],
    ) -> None:
        if self.log_dir is None:
            return
        if self._csv_path is None or self._json_path is None:
            self._prepare_logging_files()
        if self._csv_path is None or self._json_path is None:
            return
        overall_anchor_empty = (
            float(sum(anchor_empty_rate.values()) / max(len(anchor_empty_rate), 1))
            if anchor_empty_rate
            else 0.0
        )
        ratio_map = {int(k): float(v["ratio"]) for k, v in level_stats.items()}
        selected_map = {int(k): int(v) for k, v in selected_counts.items()}
        total_map = {int(k): int(v) for k, v in level_totals.items()}
        for lvl, stat in level_stats.items():
            key = int(lvl)
            if key not in selected_map:
                selected_map[key] = int(stat.get("selected", 0))
            if key not in total_map:
                total_map[key] = int(stat.get("total", 0))
        record = {
            "iteration": int(iteration),
            "delta": float(delta_val),
            "interval": int(interval_val),
            "num_gaussians": int(num_gaussians),
            "kappa_total": int(kappa_total),
            "keep_ratio": float(self.keep_ratio) if self.keep_ratio is not None else None,
            "kappa_descriptor": kappa_descriptor,
            "anchor_m_min": int(self.anchor_m_min),
            "age_grace_iters": int(self.age_grace_iters),
            "diff_norm": float(diff_norm),
            "anchor_empty_rate": overall_anchor_empty,
            "kappa_per_level": {int(k): int(v) for k, v in kappa_map.items()},
            "selected_ratio_per_level": ratio_map,
            "selected_counts_per_level": selected_map,
            "candidate_counts_per_level": total_map,
            "anchor_empty_rate_per_level": {
                int(k): float(v) for k, v in anchor_empty_rate.items()
            },
            "thresholds_per_level": {
                int(k): {kk: float(vv) for kk, vv in vals.items()}
                for k, vals in thresholds.items()
            },
        }
        if self._csv_path is not None:
            csv_record = record.copy()
            csv_record["kappa_per_level"] = json.dumps(record["kappa_per_level"])
            csv_record["selected_ratio_per_level"] = json.dumps(
                record["selected_ratio_per_level"]
            )
            csv_record["selected_counts_per_level"] = json.dumps(
                record["selected_counts_per_level"]
            )
            csv_record["candidate_counts_per_level"] = json.dumps(
                record["candidate_counts_per_level"]
            )
            csv_record["anchor_empty_rate_per_level"] = json.dumps(
                record["anchor_empty_rate_per_level"]
            )
            csv_record["thresholds_per_level"] = json.dumps(
                record["thresholds_per_level"]
            )
            with self._csv_path.open("a", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self._csv_fields)
                if not self._csv_header_written:
                    writer.writeheader()
                    self._csv_header_written = True
                writer.writerow(csv_record)
        if self._json_path is not None:
            with self._json_path.open("a", encoding="utf-8") as json_file:
                json_file.write(json.dumps(record) + "\n")

    def _kappa_per_level(self, unique_levels: Tensor) -> Dict[int, int]:
        result: Dict[int, int] = {}
        for lvl in unique_levels:
            key = int(lvl.item())
            result[key] = int(self.level_quota.get(key, self.anchor_m_min))
        return result

    def _kappa_descriptor(self, kappa_total: int) -> str:
        if self.kappa_total is not None:
            return f"kappa_total={int(self.kappa_total)}"
        if self.keep_ratio is not None:
            return (
                f"keep_ratio={float(self.keep_ratio):.6f}"
                f" (kappa_total={int(kappa_total)})"
            )
        return f"kappa_total={int(kappa_total)}"

    def _anchor_empty_rate_per_level(
        self,
        anchors: Tensor,
        selected: Tensor,
        levels: Tensor,
        unique_levels: Tensor,
    ) -> Dict[int, float]:
        rates: Dict[int, float] = {}
        if unique_levels.numel() == 0:
            return rates
        for lvl in unique_levels:
            level_mask = levels == lvl
            anchor_subset = anchors[level_mask]
            if anchor_subset.numel() == 0:
                rates[int(lvl.item())] = 0.0
                continue
            unique_anchors = torch.unique(anchor_subset)
            if unique_anchors.numel() == 0:
                rates[int(lvl.item())] = 0.0
                continue
            empty_count = 0
            total_count = unique_anchors.numel()
            for anchor_value in unique_anchors:
                anchor_mask = (anchors == anchor_value) & level_mask
                if not bool(selected[anchor_mask].any()):
                    empty_count += 1
            rates[int(lvl.item())] = float(empty_count) / float(total_count)
        return rates

    def _quantiles_per_level(
        self,
        combined: Tensor,
        levels: Tensor,
        selected: Tensor,
        unique_levels: Tensor,
    ) -> Dict[int, Dict[str, float]]:
        thresholds: Dict[int, Dict[str, float]] = {}
        if unique_levels.numel() == 0:
            return thresholds
        quantiles = torch.tensor([0.5, 0.9, 0.99], device=combined.device)
        for lvl in unique_levels:
            level_mask = levels == lvl
            level_selected = level_mask & selected
            values = combined[level_selected]
            if values.numel() == 0:
                thresholds[int(lvl.item())] = {"p50": 0.0, "p90": 0.0, "p99": 0.0}
                continue
            q_values = torch.quantile(values, quantiles).tolist()
            thresholds[int(lvl.item())] = {
                "p50": float(q_values[0]),
                "p90": float(q_values[1]),
                "p99": float(q_values[2]),
            }
        return thresholds

