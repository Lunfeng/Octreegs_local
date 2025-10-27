#!/usr/bin/env python3
import argparse
import logging
import os
import shutil
from typing import Optional

import numpy as np
import torch

from arguments import ModelParams, OptimizationParams
from scene import Scene, GaussianModel


LOGGER = logging.getLogger("prune_and_save")


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
            raise FileNotFoundError(f"Unable to locate cfg_args near {path}")
        cur = parent


def _load_cfg_namespace(model_root: str):
    with open(os.path.join(model_root, "cfg_args"), "r", encoding="utf-8") as fp:
        return eval(fp.read())


def _load_gaussians(model_root: str, cfg_ns):
    parser = argparse.ArgumentParser(add_help=False)
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)

    args = argparse.Namespace(**vars(cfg_ns))
    model_args = lp.extract(args)
    opt_args = op.extract(args)
    model_args.model_path = model_root

    gaussians = GaussianModel(
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

    gaussians.checkpoint_dir = model_root
    Scene(
        model_args,
        gaussians,
        load_iteration=-1,
        shuffle=False,
        resolution_scales=model_args.resolution_scales,
        logger=LOGGER,
    )

    gaussians.training_setup(opt_args)
    gaussians.maybe_load_vq_anchor_feat(model_root)
    return gaussians, opt_args


def _get_latest_checkpoint(model_root: str) -> Optional[str]:
    candidates = [
        os.path.join(model_root, f)
        for f in os.listdir(model_root)
        if f.endswith(".pth")
    ]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _apply_checkpoint(gaussians: GaussianModel, opt_args, checkpoint: str):
    LOGGER.info("Loading optimizer state from %s", checkpoint)
    payload = torch.load(checkpoint)
    if isinstance(payload, tuple) and len(payload) == 2:
        state, _ = payload
    else:
        state = payload
    gaussians.restore(state, opt_args)


def _compute_dir_size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    size = 0
    for root, _dirs, files in os.walk(path):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                size += os.path.getsize(fpath)
            except OSError:
                continue
    return size


def main():
    _configure_logging()

    parser = argparse.ArgumentParser(description="Prune anchors based on importance score")
    parser.add_argument("--ckpt", required=True, help="Checkpoint directory or file")
    parser.add_argument("--imp", required=True, help="npz file with importance score")
    parser.add_argument("--prune_ratio", type=float, required=True, help="Ratio of anchors to prune")
    parser.add_argument("--out_dir", required=True, help="Output checkpoint directory")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not 0.0 < args.prune_ratio < 1.0:
        raise ValueError("--prune_ratio must be in (0,1)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_root = _find_model_root(args.ckpt)
    cfg_ns = _load_cfg_namespace(model_root)
    gaussians, opt_args = _load_gaussians(model_root, cfg_ns)

    ckpt_file = args.ckpt if os.path.isfile(args.ckpt) else _get_latest_checkpoint(model_root)
    if ckpt_file:
        gaussians.checkpoint_dir = os.path.dirname(ckpt_file)
        _apply_checkpoint(gaussians, opt_args, ckpt_file)

    scores = np.load(args.imp)["score"]
    if scores.shape[0] != gaussians.get_anchor.shape[0]:
        raise ValueError("Importance score length does not match number of anchors")

    scores_t = torch.from_numpy(scores)
    anchor_count = scores_t.shape[0]
    prune_count = int(anchor_count * args.prune_ratio)
    LOGGER.info("Total anchors: %d, pruning %d", anchor_count, prune_count)

    sorted_idx = torch.argsort(scores_t)
    prune_idx = sorted_idx[:prune_count]
    mask_to_prune = torch.zeros(anchor_count, dtype=torch.bool)
    mask_to_prune[prune_idx] = True

    before_anchor = gaussians.get_anchor.shape[0]
    before_offsets = gaussians._offset.shape[0] * gaussians._offset.shape[1]

    gaussians.prune_anchor(mask_to_prune.cuda() if gaussians.get_anchor.is_cuda else mask_to_prune)

    after_anchor = gaussians.get_anchor.shape[0]
    after_offsets = gaussians._offset.shape[0] * gaussians._offset.shape[1]

    LOGGER.info("Anchors: %d -> %d", before_anchor, after_anchor)
    LOGGER.info("Offsets: %d -> %d", before_offsets, after_offsets)

    os.makedirs(args.out_dir, exist_ok=True)

    # Copy configuration for reproducibility
    shutil.copy2(os.path.join(model_root, "cfg_args"), os.path.join(args.out_dir, "cfg_args"))

    point_cloud_path = os.path.join(args.out_dir, "point_cloud", "iteration_0")
    os.makedirs(point_cloud_path, exist_ok=True)
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    gaussians.save_mlp_checkpoints(point_cloud_path)

    checkpoint_path = os.path.join(args.out_dir, "pruned_checkpoint.pth")
    torch.save((gaussians.capture(), 0), checkpoint_path)

    src_size = _compute_dir_size(model_root)
    dst_size = _compute_dir_size(args.out_dir)
    LOGGER.info("Disk footprint: %.2f MB -> %.2f MB", src_size / 1e6, dst_size / 1e6)


if __name__ == "__main__":
    main()
