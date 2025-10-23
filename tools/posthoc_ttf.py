"""Command line helper to launch post-hoc TTF pruning."""

from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Union

import torch
import yaml

from octreegs.engine.trainer import Trainer
from octreegs.pruning.ttf_core import TtfController

LOGGER = logging.getLogger(__name__)


def _load_yaml_dict(path: Path) -> Dict[str, Any]:
    """Load a YAML file into a dictionary."""

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of YAML file: {path}")
    return data


def _resolve_factory(spec: str) -> Callable[..., Any]:
    """Resolve a ``module:function`` specification to a callable."""

    if ":" not in spec:
        raise ValueError("Factory path must be provided as 'module:function'")
    module_name, func_name = spec.split(":", maxsplit=1)
    module = __import__(module_name, fromlist=[func_name])
    factory = getattr(module, func_name)
    if not callable(factory):
        raise TypeError(f"Resolved attribute {func_name!r} from {module_name!r} is not callable")
    return factory


def _invoke_factory(
    factory: Callable[..., Any],
    scene_cfg: Dict[str, Any],
    config_path: Path,
    ckpt_path: Path,
    ckpt_data: Any,
    ttf_cfg: Dict[str, Any],
    ttf_cfg_path: Path,
    device: Union[str, torch.device],
) -> Any:
    """Call ``factory`` with best-effort keyword matching."""

    signature = inspect.signature(factory)
    values: Dict[str, Any] = {
        "config": scene_cfg,
        "cfg": scene_cfg,
        "scene_cfg": scene_cfg,
        "config_data": scene_cfg,
        "config_path": config_path,
        "cfg_path": config_path,
        "config_file": str(config_path),
        "ttf_cfg": ttf_cfg,
        "ttf_config": ttf_cfg,
        "ttf": ttf_cfg,
        "ttf_cfg_path": ttf_cfg_path,
        "ttf_config_path": ttf_cfg_path,
        "ttf_cfg_file": str(ttf_cfg_path),
        "ckpt_path": ckpt_path,
        "checkpoint_path": ckpt_path,
        "ckpt": ckpt_path,
        "checkpoint": ckpt_path,
        "ckpt_file": str(ckpt_path),
        "checkpoint_file": str(ckpt_path),
        "ckpt_data": ckpt_data,
        "checkpoint_data": ckpt_data,
        "device": device,
        "device_str": str(device),
    }
    kwargs: Dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if name in values:
            kwargs[name] = values[name]
        elif parameter.default is inspect._empty:
            raise TypeError(
                f"Factory {factory.__module__}.{factory.__name__} requires argument {name!r}"
            )
    return factory(**kwargs)


def _materialise_controller(
    result: Any,
    ttf_cfg: Dict[str, Any],
    device: Union[str, torch.device],
) -> TtfController:
    """Convert a factory result into a :class:`TtfController` instance."""

    if isinstance(result, TtfController):
        return TtfController(
            result.model,
            result.renderer,
            result.train_step_fn,
            result.eval_step_fn,
            ttf_cfg,
            device=device,
        )
    if isinstance(result, Trainer):
        controller = TtfController(
            result.model,
            result.renderer,
            result._ttf_train_step,
            result.evaluate,
            ttf_cfg,
            device=device,
        )
        result.ttf = controller
        return controller
    if isinstance(result, dict):
        model = result.get("model")
        renderer = result.get("renderer")
        train_step = result.get("train_step_fn") or result.get("train_step")
        eval_step = result.get("eval_step_fn") or result.get("eval_step")
        if model is None or renderer is None or train_step is None or eval_step is None:
            raise TypeError("Factory dictionary output must include model/renderer/train_step_fn/eval_step_fn")
        return TtfController(model, renderer, train_step, eval_step, ttf_cfg, device=device)
    if isinstance(result, (tuple, list)) and len(result) >= 4:
        model, renderer, train_step, eval_step = result[:4]
        return TtfController(model, renderer, train_step, eval_step, ttf_cfg, device=device)
    raise TypeError("Unable to convert factory output into a TtfController instance")


def _restore_checkpoint(controller: TtfController, ckpt_data: Any) -> None:
    """Load state dict and prune mask from checkpoint when possible."""

    if not isinstance(ckpt_data, dict):
        return
    state_dict = None
    for key in ("state_dict", "model", "model_state_dict"):
        if key in ckpt_data and isinstance(ckpt_data[key], dict):
            state_dict = ckpt_data[key]
            break
    if state_dict is not None:
        controller.model.load_state_dict(state_dict, strict=False)
    mask = ckpt_data.get("prune_mask") or ckpt_data.get("mask")
    if mask is not None:
        prune_mask = torch.as_tensor(mask, dtype=torch.bool, device=controller.prune_mask.device)
        if prune_mask.shape == controller.prune_mask.shape:
            controller.prune_mask.copy_(prune_mask)
            controller.model.set_prune_mask(controller.prune_mask)


def _write_metrics(path: Path, metrics: Dict[str, float]) -> None:
    """Persist metrics to ``path`` in key=value format."""

    with path.open("w", encoding="utf-8") as handle:
        for key in sorted(metrics):
            value = metrics[key]
            handle.write(f"{key}: {value:.6f}\n")


def _clean_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Convert evaluation metrics to scalar floats."""

    cleaned: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                cleaned[key] = float(value.item())
            else:
                continue
        elif isinstance(value, (int, float)):
            cleaned[key] = float(value)
    return cleaned


def main() -> None:
    """Entry-point hooking CLI arguments to the pruning controller."""

    parser = argparse.ArgumentParser(description="Run post-hoc TTF pruning")
    parser.add_argument("--config", type=Path, required=True, help="Scene or trainer configuration YAML")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint containing the model state")
    parser.add_argument("--ttf-cfg", type=Path, required=True, help="TTF configuration YAML")
    parser.add_argument(
        "--ttf-preset",
        type=str,
        default="balanced",
        choices=["mild", "balanced", "aggressive"],
        help="Preset defined in the TTF configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <ckpt_parent>/ttf_posthoc",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for controller instantiation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    scene_cfg = _load_yaml_dict(args.config)
    ttf_cfg = _load_yaml_dict(args.ttf_cfg)

    ttf_section = scene_cfg.get("ttf") or {}
    if not isinstance(ttf_section, dict):
        raise ValueError("Scene configuration must contain a 'ttf' mapping")
    factory_spec = ttf_section.get("controller_factory") or scene_cfg.get("controller_factory")
    if not factory_spec:
        raise ValueError("Configuration must define 'controller_factory' under the 'ttf' section")

    factory = _resolve_factory(factory_spec)
    ckpt_data = torch.load(args.ckpt, map_location=args.device)
    controller_obj = _invoke_factory(
        factory,
        scene_cfg=scene_cfg,
        config_path=args.config,
        ckpt_path=args.ckpt,
        ckpt_data=ckpt_data,
        ttf_cfg=ttf_cfg,
        ttf_cfg_path=args.ttf_cfg,
        device=args.device,
    )
    controller = _materialise_controller(controller_obj, ttf_cfg, device=args.device)
    _restore_checkpoint(controller, ckpt_data)

    output_dir = args.output_dir or (args.ckpt.parent / "ttf_posthoc")
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting post-hoc TTF with preset '%s'", args.ttf_preset)
    controller.model.freeze_growth(True)
    try:
        metrics = controller.run_posthoc_phase(preset=args.ttf_preset)
    finally:
        controller.model.freeze_growth(False)
    if not isinstance(metrics, dict):
        LOGGER.warning("run_posthoc_phase returned non-dict metrics (%s); ignoring", type(metrics))
        metrics = {}
    cleaned_metrics = _clean_metrics(metrics)
    LOGGER.info("Post-hoc TTF completed: %s", cleaned_metrics)

    metrics_path = output_dir / f"final_metrics_{args.ttf_preset}.txt"
    model_path = output_dir / f"model_ttf_{args.ttf_preset}.pth"
    _write_metrics(metrics_path, cleaned_metrics)
    torch.save(
        {
            "state_dict": controller.model.state_dict(),
            "prune_mask": controller.prune_mask.detach().cpu(),
            "ttf_config": ttf_cfg,
            "preset": args.ttf_preset,
        },
        model_path,
    )
    LOGGER.info("Saved metrics to %s and model to %s", metrics_path, model_path)


if __name__ == "__main__":
    main()
