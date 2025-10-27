#!/usr/bin/env python3
import argparse
import logging
import os
from typing import Tuple

import numpy as np
import torch

from arguments import ModelParams
from scene import Scene, GaussianModel


LOGGER = logging.getLogger("vq_anchor_feat")


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
    args = argparse.Namespace(**vars(cfg_ns))
    model_args = lp.extract(args)
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
    gaussians.maybe_load_vq_anchor_feat(model_root)
    return gaussians, model_args


def _gather_quant_subset(scores: torch.Tensor, ratio: float) -> torch.Tensor:
    num = scores.shape[0]
    count = int(num * ratio)
    if count == 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)
    order = torch.argsort(scores)
    return order[:count]


def _nearest_assign(feats: torch.Tensor, codebook: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
    assignments = []
    for start in range(0, feats.shape[0], chunk):
        end = min(start + chunk, feats.shape[0])
        chunk_feats = feats[start:end]
        dist = torch.cdist(chunk_feats, codebook)
        assignments.append(dist.argmin(dim=1))
    return torch.cat(assignments, dim=0)


def _ema_update(
    feats: torch.Tensor,
    assignments: torch.Tensor,
    codebook: torch.Tensor,
    ema_cluster_size: torch.Tensor,
    ema_codebook: torch.Tensor,
    decay: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    one_hot = torch.nn.functional.one_hot(assignments, num_classes=codebook.shape[0]).to(feats.dtype)
    counts = one_hot.sum(dim=0)
    sums = one_hot.transpose(0, 1) @ feats

    ema_cluster_size = decay * ema_cluster_size + (1.0 - decay) * counts
    ema_codebook = decay * ema_codebook + (1.0 - decay) * sums

    n = ema_codebook / ema_cluster_size.unsqueeze(1).clamp(min=1e-6)
    codebook = n
    return codebook, ema_cluster_size, ema_codebook


def _run_vq(
    feats: torch.Tensor,
    codebook_size: int,
    phase1_steps: int,
    phase2_steps: int,
    ema_decay: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = feats.device

    unique = torch.unique(feats, dim=0)
    if unique.shape[0] < codebook_size:
        LOGGER.warning("Reducing codebook size from %d to %d", codebook_size, unique.shape[0])
        codebook_size = unique.shape[0]

    perm = torch.randperm(unique.shape[0], device=device)
    codebook = unique[perm[:codebook_size]].clone()
    ema_codebook = codebook.clone()
    ema_cluster_size = torch.ones(codebook_size, device=device)

    assignments = _nearest_assign(feats, codebook)

    for _ in range(phase1_steps):
        codebook, ema_cluster_size, ema_codebook = _ema_update(
            feats,
            assignments,
            codebook,
            ema_cluster_size,
            ema_codebook,
            ema_decay,
        )

    for _ in range(phase2_steps):
        assignments = _nearest_assign(feats, codebook)
        codebook, ema_cluster_size, ema_codebook = _ema_update(
            feats,
            assignments,
            codebook,
            ema_cluster_size,
            ema_codebook,
            ema_decay,
        )

    return codebook, assignments


def main():
    _configure_logging()

    parser = argparse.ArgumentParser(description="Selective VQ for anchor features")
    parser.add_argument("--ckpt", required=True, help="Checkpoint directory or file")
    parser.add_argument("--imp", required=True, help="Importance score npz path")
    parser.add_argument("--quant_ratio", type=float, required=True)
    parser.add_argument("--codebook_size", type=int, default=8192)
    parser.add_argument("--ema_decay", type=float, default=0.9)
    parser.add_argument("--commit", type=float, default=1.0)
    parser.add_argument("--phase1_steps", type=int, default=2000)
    parser.add_argument("--phase2_steps", type=int, default=2000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not 0.0 < args.quant_ratio <= 1.0:
        raise ValueError("--quant_ratio must be in (0,1]")

    model_root = _find_model_root(args.ckpt)
    cfg_ns = _load_cfg_namespace(model_root)
    gaussians, _ = _load_gaussians(model_root, cfg_ns)

    scores = np.load(args.imp)["score"]
    if scores.shape[0] != gaussians.get_anchor.shape[0]:
        raise ValueError("Importance score length mismatch")

    scores_t = torch.from_numpy(scores).to(gaussians.get_anchor.device)
    feat = gaussians.get_anchor_feat.detach().clone()

    quant_indices = _gather_quant_subset(scores_t, args.quant_ratio)
    if quant_indices.numel() == 0:
        LOGGER.info("No anchors selected for quantization, skipping")
        keep_mask = torch.ones_like(scores_t, dtype=torch.bool)
        np.savez(args.out, codebook=np.empty((0, feat.shape[1]), dtype=np.float32), indices=np.empty((0,), dtype=np.int32), keep_mask=keep_mask.cpu().numpy(), feat_dim=feat.shape[1])
        return

    subset_feat = feat[quant_indices].to(torch.float32)
    codebook, assignments = _run_vq(
        subset_feat,
        min(args.codebook_size, subset_feat.shape[0]),
        args.phase1_steps,
        args.phase2_steps,
        args.ema_decay,
    )

    quantized = codebook[assignments]
    error = torch.mean((subset_feat - quantized) ** 2).item()
    LOGGER.info("Quantization MSE: %.6f", error)

    keep_mask = torch.ones(feat.shape[0], dtype=torch.bool, device=feat.device)
    keep_mask[quant_indices] = False

    codebook_np = codebook.cpu().numpy().astype(np.float16)
    indices_np = assignments.cpu().numpy().astype(np.int32)
    keep_mask_np = keep_mask.cpu().numpy()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        args.out,
        codebook=codebook_np,
        indices=indices_np,
        keep_mask=keep_mask_np,
        feat_dim=feat.shape[1],
        commit=args.commit,
        mse=error,
    )

    LOGGER.info(
        "Saved VQ artifacts to %s (codebook: %d x %d, quantized anchors: %d)",
        args.out,
        codebook_np.shape[0],
        codebook_np.shape[1] if codebook_np.size else feat.shape[1],
        quant_indices.numel(),
    )


if __name__ == "__main__":
    main()
