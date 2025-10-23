from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import torch.distributed as dist
except ImportError:  # pragma: no cover - torch always available in runtime
    dist = None


@dataclass
class MaskTemperatureSchedule:
    init: float
    final: float
    anneal_portion: float


class MaskSampler:
    """Sample binary keep/drop decisions using configurable straight-through estimators."""

    def __init__(
        self,
        mask_config,
        total_steps: int,
        *,
        base_seed: int = 0,
    ) -> None:
        temp_cfg = getattr(mask_config, "temp", None)
        if temp_cfg is None:
            raise ValueError("mask_config must define a temp schedule")

        self.enabled = bool(getattr(mask_config, "enabled", False))
        self.sample_trials = max(int(getattr(mask_config, "sample_trials", 1)), 1)
        sampler_mode = getattr(mask_config, "sampler", getattr(mask_config, "sampler_mode", "gumbel"))
        self.mode = str(sampler_mode).lower()
        if self.mode not in {"gumbel", "ste"}:
            raise ValueError(f"Unsupported mask sampler mode: {self.mode}")
        self.temperature = MaskTemperatureSchedule(
            init=float(getattr(temp_cfg, "init", 1.0)),
            final=float(getattr(temp_cfg, "final", 0.4)),
            anneal_portion=float(getattr(temp_cfg, "anneal_portion", 1.0)),
        )
        self.total_steps = max(int(total_steps), 1)
        self.base_seed = int(base_seed)

        self._rank = 0
        self._world = 1
        if dist is not None and dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world = max(dist.get_world_size(), 1)

    def _temperature_at_step(self, step: int) -> float:
        anneal_span = max(int(math.ceil(self.total_steps * self.temperature.anneal_portion)), 1)
        progress = min(max(step, 0) / anneal_span, 1.0)
        return self.temperature.init + (self.temperature.final - self.temperature.init) * progress

    def _gumbel_noise(
        self,
        shape: torch.Size,
        step: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        rng_offset: int = 0,
    ) -> torch.Tensor:
        generator = torch.Generator(device=device)
        seed = self.base_seed + (step + rng_offset) * self._world + self._rank
        generator.manual_seed(seed)
        uniform = torch.rand(shape, generator=generator, device=device, dtype=dtype)
        eps = torch.finfo(dtype).eps
        uniform = uniform.clamp_(eps, 1.0 - eps)
        return -torch.log(-torch.log(uniform))

    def sample(
        self,
        keep_logits: torch.Tensor,
        drop_logits: torch.Tensor,
        step: int,
        newborn_cycles: Optional[torch.Tensor] = None,
        *,
        rng_offset: int = 0,
        return_trials: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Draw hard binary masks aligned with the provided logits."""
        if keep_logits.numel() == 0:
            empty_mask = torch.empty(0, dtype=torch.uint8, device=keep_logits.device)
            empty_keep = torch.empty(0, dtype=keep_logits.dtype, device=keep_logits.device)
            empty_probs = torch.empty(0, 2, dtype=keep_logits.dtype, device=keep_logits.device)
            return empty_mask, empty_keep, empty_probs

        if keep_logits.shape != drop_logits.shape:
            raise ValueError("keep and drop logits must share the same shape")

        if not torch.isfinite(keep_logits).all():
            raise RuntimeError("Non-finite values detected in mask_logit_keep")
        if not torch.isfinite(drop_logits).all():
            raise RuntimeError("Non-finite values detected in mask_logit_drop")

        keep_flat = keep_logits.reshape(-1, 1)
        drop_flat = drop_logits.reshape(-1, 1)
        logits = torch.cat([keep_flat, drop_flat], dim=-1)

        tau = self._temperature_at_step(step)

        if self.mode == "gumbel":
            if self.sample_trials <= 1:
                gumbel = self._gumbel_noise(logits.shape, step, logits.device, logits.dtype, rng_offset=rng_offset)
                scaled = (logits + gumbel) / tau
                probs_full = F.softmax(scaled, dim=-1)
                hard_index = probs_full.argmax(dim=-1, keepdim=True)
                hard = torch.zeros_like(probs_full).scatter_(dim=-1, index=hard_index, value=1.0)
                straight_through = hard - probs_full.detach() + probs_full
                keep_values = straight_through[..., 0]
                keep_trials = hard[..., 0].unsqueeze(0)
                probs = probs_full
            else:
                trial_shape = (self.sample_trials,) + logits.shape
                gumbel = self._gumbel_noise(trial_shape, step, logits.device, logits.dtype, rng_offset=rng_offset)
                expanded_logits = logits.unsqueeze(0).expand(self.sample_trials, -1, -1)
                scaled = (expanded_logits + gumbel) / tau
                probs_full = F.softmax(scaled, dim=-1)
                hard_index = probs_full.argmax(dim=-1, keepdim=True)
                hard = torch.zeros_like(probs_full).scatter_(dim=-1, index=hard_index, value=1.0)
                straight_through = hard - probs_full.detach() + probs_full
                keep_values = straight_through[0, ..., 0]
                keep_trials = hard[..., 0]
                probs = probs_full[0]
        else:  # Straight-through estimator without Gumbel noise
            delta = (keep_flat - drop_flat) / tau
            probs_keep = torch.sigmoid(delta)
            probs_full = torch.cat([probs_keep, 1.0 - probs_keep], dim=-1)
            hard = (probs_keep >= 0.5).to(probs_keep.dtype)
            straight_through_keep = hard - probs_keep.detach() + probs_keep
            keep_values = straight_through_keep.reshape(-1)
            probs = probs_full.reshape(-1, 2)
            hard_trials = hard.reshape(1, -1)
            keep_trials = hard_trials.expand(self.sample_trials, -1)

        if newborn_cycles is not None and newborn_cycles.numel() > 0:
            newborn = newborn_cycles.reshape(-1).to(dtype=keep_values.dtype)
            keep_values = torch.where(newborn > 0, torch.ones_like(keep_values), keep_values)
            newborn_expanded = newborn.unsqueeze(0)
            keep_trials = torch.where(
                newborn_expanded > 0,
                torch.ones_like(keep_trials, dtype=keep_trials.dtype),
                keep_trials,
            )

        mask_bits = keep_values.to(torch.uint8)
        mask_bits = mask_bits.contiguous()
        probs = probs.contiguous()
        if return_trials:
            return mask_bits, keep_values, probs, keep_trials.to(torch.uint8).contiguous()
        return mask_bits, keep_values, probs


__all__ = ["MaskSampler"]
