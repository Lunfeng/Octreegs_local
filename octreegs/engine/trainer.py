"""Training engine with optional TTF pruning integration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from octreegs.pruning.ttf_core import TtfConfig, TtfController


@dataclass
class TrainerTtfOptions:
    enable_online: bool = False
    enable_posthoc: bool = False
    preset: str = "balanced"
    gamma_iter: float = 0.275
    alpha_q: float = 0.6
    grad_q: float = 0.6
    short_ft_iters: int = 600


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
        ttf_options: Optional[TrainerTtfOptions] = None,
        train_step_impl: Optional[Callable[[bool, float], Dict]] = None,
        eval_step_impl: Optional[Callable[[bool, bool], Dict]] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.renderer = renderer
        self.cfg = cfg
        self._train_step_impl = train_step_impl
        self._eval_step_impl = eval_step_impl
        self.ttf_options = ttf_options or TrainerTtfOptions()
        self.ttf: Optional[TtfController] = None
        if isinstance(ttf_options, TrainerTtfOptions):
            ttf_cfg = getattr(cfg, "ttf", None)
            if isinstance(ttf_cfg, dict):
                ttf_cfg = TtfConfig(**ttf_cfg)
            if isinstance(ttf_cfg, TtfConfig):
                self.ttf = TtfController(model, self.step, self.evaluate, ttf_cfg)

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

    def evaluate(self, return_rgb: bool = False, tiles: bool = False) -> Dict:
        if self._eval_step_impl is None:
            raise NotImplementedError("Trainer.evaluate requires an eval_step_impl callable")
        return self._eval_step_impl(return_rgb, tiles)

    # ------------------------------------------------------------------
    # Integration helpers
    # ------------------------------------------------------------------
    def on_growth_window_end(self) -> None:
        if self.ttf is None or not self.ttf_options.enable_online:
            return
        schedule = {
            "rounds": 1,
            "gamma_iter": self.ttf_options.gamma_iter,
            "alpha_q": self.ttf_options.alpha_q,
            "grad_q": self.ttf_options.grad_q,
            "short_ft_iters": self.ttf_options.short_ft_iters,
        }
        self.ttf.run_online_phase(schedule)

    def run_posthoc(self) -> None:
        if self.ttf is None or not self.ttf_options.enable_posthoc:
            return
        self.ttf.run_posthoc_phase(self.ttf_options.preset)
