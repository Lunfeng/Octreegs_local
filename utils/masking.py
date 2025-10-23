"""
Masking utilities for probabilistic pruning in OctreeGS.
Implements Gumbel-Softmax binary sampling and temperature scheduling.
"""

import torch
import math
from typing import Tuple


def gumbel_softmax_binary(
    logits_keep: torch.Tensor,
    logits_drop: torch.Tensor,
    tau: float,
    hard: bool = True,
    eps: float = 1e-9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Binary Gumbel-Softmax sampling for mask generation.
    
    Args:
        logits_keep: [N, 1] logits for keeping gaussians
        logits_drop: [N, 1] logits for dropping gaussians
        tau: temperature parameter
        hard: if True, return binary masks with straight-through estimator
        eps: small constant for numerical stability
    
    Returns:
        p_keep: [N, 1] soft probabilities of keeping
        M: [N, 1] binary mask (0 or 1) if hard=True, soft probabilities otherwise
    """
    # Sample Gumbel noise
    u1 = torch.rand_like(logits_keep)
    u2 = torch.rand_like(logits_drop)
    g1 = -torch.log(-torch.log(u1 + eps) + eps)
    g2 = -torch.log(-torch.log(u2 + eps) + eps)
    
    # Add Gumbel noise and scale by temperature
    y1 = (logits_keep + g1) / tau
    y2 = (logits_drop + g2) / tau
    
    # Concatenate and apply softmax
    cat = torch.cat([y1, y2], dim=-1)  # [N, 2]
    probs = torch.softmax(cat, dim=-1)  # [N, 2]
    p_keep = probs[..., :1]  # [N, 1]
    
    if hard:
        # Create hard binary mask
        M = torch.zeros_like(p_keep)
        M.scatter_(-1, torch.argmax(probs, dim=-1, keepdim=True)[..., :1], 1.0)
        # Straight-through estimator: use hard mask for forward, soft for backward
        M = (M - p_keep).detach() + p_keep
        # Convert to strict binary
        M = (M > 0.5).float()
    else:
        M = p_keep
    
    return p_keep, M


class TempScheduler:
    """Temperature scheduler for Gumbel-Softmax annealing."""
    
    def __init__(self, t_start: float = 1.0, t_end: float = 0.4, total_steps: int = 30000):
        """
        Initialize temperature scheduler with linear annealing.
        
        Args:
            t_start: starting temperature
            t_end: ending temperature
            total_steps: total number of training steps
        """
        self.t_start = t_start
        self.t_end = t_end
        self.total = max(1, total_steps)
    
    def value(self, step: int) -> float:
        """Get temperature value at given step."""
        s = min(max(step, 0), self.total)
        return self.t_start + (self.t_end - self.t_start) * (s / self.total)
