#!/usr/bin/env python3
import argparse
import logging
import os
import time
from typing import List

import numpy as np
import torch

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import prefilter_voxel
from gaussian_renderer.gaussian_count_ogs import count_render_for_view
from gaussian_renderer import generate_neural_gaussians
from scene import Scene, GaussianModel


LOGGER = logging.getLogger("compute_imp_score")


def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )


def _find_model_root(path: str) -> str:
    cur = os.path.abspath(path)
    if os.path.isfile(cur):
        cur = os.path.dirname(cur)
    while True:
        cfg = os.path.join(cur, "cfg_args")
        if os.path.exists(cfg):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise FileNotFoundError(f"Failed to locate cfg_args near {path}")
        cur = parent


def _load_cfg_namespace(model_root: str):
    cfg_path = os.path.join(model_root, "cfg_args")
    with open(cfg_path, "r", encoding="utf-8") as fp:
        content = fp.read()
    return eval(content)


def _select_cameras(cams: List, subsample: int) -> List:
    if subsample <= 1:
        return cams
    return [cam for idx, cam in enumerate(cams) if idx % subsample == 0]


def _prepare_background(dataset_args) -> torch.Tensor:
    if dataset_args.random_background:
        color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    elif dataset_args.white_background:
        color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    else:
        color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    return color.cuda() if torch.cuda.is_available() else color


def _resolve_checkpoint_path(model_root: str, ckpt_arg: str) -> Optional[str]:
    if ckpt_arg and os.path.isfile(ckpt_arg):
        return ckpt_arg
    # Prefer explicit checkpoint argument first
    if ckpt_arg and os.path.isdir(ckpt_arg):
        # Look for torch checkpoint in provided directory
        candidates = [
            os.path.join(ckpt_arg, f)
            for f in os.listdir(ckpt_arg)
            if f.endswith(".pth")
        ]
        if candidates:
            return sorted(candidates)[-1]
    # Fallback: use latest chkpnt in model root if available
    chkpnt_dir = model_root
    checkpoints = [
        os.path.join(chkpnt_dir, f)
        for f in os.listdir(chkpnt_dir)
        if f.startswith("chkpnt") and f.endswith(".pth")
    ]
    if checkpoints:
        return sorted(checkpoints)[-1]
    return None


def _load_scene_and_model(model_root: str, device: torch.device, cfg_ns):
    parser = argparse.ArgumentParser(add_help=False)
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    args = argparse.Namespace(**vars(cfg_ns))
    model_args = lp.extract(args)
    opt_args = op.extract(args)
    pipe_args = pp.extract(args)

    model_args.model_path = model_root
    dataset_gaussians = GaussianModel(
        model_args.feat_dim,
        model_args.n_offsets,
        model_args.fork,
        model_args.use_feat_bank,
        model_args.appearance_dim,
        model_args.add_opacity_dist,
        model_args.add_cov_dist,
        model_args.add_color_dist,
        model_args.add_level,
        model_args.visible_threshold,
        model_args.dist2level,
        model_args.base_layer,
        model_args.progressive,
        model_args.extend,
    )

    dataset_gaussians.checkpoint_dir = model_root
    load_iteration = -1
    scene = Scene(
        model_args,
        dataset_gaussians,
        load_iteration=load_iteration,
        shuffle=False,
        resolution_scales=model_args.resolution_scales,
        logger=LOGGER,
    )

    checkpoint_path = _resolve_checkpoint_path(model_root, getattr(args, "start_checkpoint", None))
    if checkpoint_path:
        dataset_gaussians.checkpoint_dir = os.path.dirname(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location=device)
        if isinstance(payload, tuple) and len(payload) == 2:
            state, _ = payload
        else:
            state = payload
        dataset_gaussians.restore(state, opt_args)

    dataset_gaussians.maybe_load_vq_anchor_feat(model_root)

    background = _prepare_background(model_args)
    return scene, dataset_gaussians, pipe_args, background


def main():
    _configure_logging()

    parser = argparse.ArgumentParser(description="Compute global importance score")
    parser.add_argument("--ckpt", required=True, help="Checkpoint directory or file")
    parser.add_argument("--out", required=True, help="Output npz path")
    parser.add_argument("--subsample", type=int, default=1, help="Subsample interval for cameras")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id; -1 for auto")
    args = parser.parse_args()

    if args.subsample < 1:
        raise ValueError("--subsample must be >= 1")

    model_root = _find_model_root(args.ckpt)
    cfg_ns = _load_cfg_namespace(model_root)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")

    LOGGER.info("Using device %s", device)

    scene, gaussians, pipe_args, background = _load_scene_and_model(model_root, device, cfg_ns)

    train_cameras = scene.getTrainCameras()
    selected_cams = _select_cameras(train_cameras, args.subsample)
    LOGGER.info("Total cameras: %d, selected: %d", len(train_cameras), len(selected_cams))

    anchor_count = gaussians.get_anchor.shape[0]
    score_anchor = torch.zeros(anchor_count, dtype=torch.float64, device=device)

    start_time = time.time()
    processed = 0
    for cam in selected_cams:
        try:
            gaussians.set_anchor_mask(cam.camera_center, 0, cam.resolution_scale)
            prefilter_voxel(cam, gaussians, pipe_args, background)

            hits, offset2anchor = count_render_for_view(
                cam,
                gaussians,
                background,
                device=device,
            )

            if hits.numel() == 0:
                continue

            generated = generate_neural_gaussians(
                cam,
                gaussians,
                visible_mask=getattr(gaussians, "_anchor_mask", None),
                is_training=True,
            )
            _xyz, _color, opacity, scaling, _rot, _neural, mask = generated

            opacity = opacity.view(-1).to(device)
            scaling = scaling[:, :3].to(device)
            volume = scaling.prod(dim=1)
            hit_float = hits.to(device=device, dtype=opacity.dtype)
            score_offset = hit_float * opacity * volume

            if offset2anchor is not None:
                offset2anchor = offset2anchor.to(device)
                score_anchor.scatter_add_(0, offset2anchor.long(), score_offset.double())
            else:
                n_offsets = gaussians.n_offsets
                mask_full = mask
                total_offsets = mask_full.numel()
                anchor_indices = torch.arange(total_offsets // n_offsets, device=device)
                anchor_indices = anchor_indices.repeat_interleave(n_offsets)
                anchor_indices = anchor_indices[mask_full.to(device)]
                score_anchor.scatter_add_(0, anchor_indices.long(), score_offset.double())

            processed += 1
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Failed to process camera %s: %s", cam.image_name, exc)

    elapsed = time.time() - start_time
    LOGGER.info("Processed %d / %d cameras in %.2f seconds", processed, len(selected_cams), elapsed)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    score_cpu = score_anchor.cpu().numpy().astype(np.float64)
    np.savez(args.out, score=score_cpu)

    sorted_idx = np.argsort(score_cpu)
    top = sorted_idx[::-1][:10]
    bottom = sorted_idx[:10]
    LOGGER.info("Top 10 scores: %s", score_cpu[top])
    LOGGER.info("Bottom 10 scores: %s", score_cpu[bottom])
    LOGGER.info("Saved importance scores to %s", args.out)


if __name__ == "__main__":
    main()
