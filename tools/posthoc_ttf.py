from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from arguments import ModelParams, PipelineParams, get_combined_args
from octreegs.models.octree_model import OctreeGSModel
from octreegs.pruning.ttf_core import TtfController
from scene import Scene
from utils.general_utils import safe_state


def load_ttf_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"TTF configuration not found at {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-hoc TTF pruning.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, help="Checkpoint iteration to load.")
    parser.add_argument("--preset", default="balanced", choices=["mild", "balanced", "aggressive"], help="TTF preset")
    parser.add_argument("--ttf-config", default="configs/ttf.yaml", help="Path to the TTF configuration file")
    parser.add_argument("--output", default=None, help="Optional metrics output path")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    cfg = load_ttf_config(args.ttf_config)

    dataset = model.extract(args)
    pipeline.extract(args)  # Unused but keeps API compatibility

    gaussians = OctreeGSModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.fork,
        dataset.use_feat_bank,
        dataset.appearance_dim,
        dataset.add_opacity_dist,
        dataset.add_cov_dist,
        dataset.add_color_dist,
        dataset.add_level,
        dataset.visible_threshold,
        dataset.dist2level,
        dataset.base_layer,
        dataset.progressive,
        dataset.extend,
    )
    Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, resolution_scales=dataset.resolution_scales)
    gaussians.eval()

    controller = TtfController(gaussians, trainer=None, cfg=cfg)
    before = gaussians.num_gaussians()
    controller.run_posthoc_phase(preset=args.preset)
    after = gaussians.num_gaussians()

    metrics = {
        "preset": args.preset,
        "iteration": args.iteration,
        "num_gaussians_before": int(before),
        "num_gaussians_after": int(after),
    }

    output_path = args.output
    if output_path is None:
        output_dir = Path(dataset.model_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"final_metrics_{args.preset}.txt"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"TTF post-hoc pruning completed. Metrics saved to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()

