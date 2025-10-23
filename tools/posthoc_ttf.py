"""Command line helper to launch post-hoc TTF pruning."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Callable, Dict

import yaml

from octreegs.pruning.ttf_core import TtfConfig


def _load_config(path: Path, preset: str) -> TtfConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    common = data.get("common", {})
    preset_cfg = data.get(preset, {})
    merged: Dict = {**common, **preset_cfg}
    return TtfConfig(**merged)


def _load_factory(factory_path: str) -> Callable:
    if ":" not in factory_path:
        raise ValueError("Factory path must be in 'module:function' format")
    module_name, func_name = factory_path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    factory = getattr(module, func_name)
    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-hoc TTF pruning")
    parser.add_argument("--config", type=Path, default=Path("configs/ttf.yaml"))
    parser.add_argument("--preset", type=str, default="balanced", choices=["mild", "balanced", "aggressive"])
    parser.add_argument("--controller-factory", type=str, required=True,
                        help="Python entrypoint returning a configured TtfController (module:function)")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--gamma-iter", type=float, default=None)
    parser.add_argument("--short-iters", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_config(args.config, args.preset)
    if args.rounds is not None:
        cfg.rounds = args.rounds
    if args.gamma_iter is not None:
        cfg.gamma_iter = args.gamma_iter
    if args.short_iters is not None:
        cfg.short_ft_iters = args.short_iters

    factory = _load_factory(args.controller_factory)
    controller = factory(cfg)
    if controller.cfg != cfg:
        controller.cfg = cfg
    controller.run_posthoc_phase(args.preset)


if __name__ == "__main__":
    main()
