#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import csv
import numpy as np
import math
import configparser

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')


import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, network_gui
from pruning.mask_sampler import MaskSampler
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, PruningParams
from torch.nn.utils import clip_grad_norm_

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

MASK_PRESET_PATH = (Path(__file__).resolve().parent / "configs" / "pruning_mask_presets.ini").resolve()


def _extract_config_preset(argv):
    for idx, token in enumerate(argv):
        if token == "--config-preset" and idx + 1 < len(argv):
            return argv[idx + 1]
        if token.startswith("--config-preset="):
            return token.split("=", 1)[1]
    return None


def _load_mask_preset_definitions():
    config = configparser.ConfigParser()
    config.optionxform = str
    try:
        read_files = config.read(MASK_PRESET_PATH, encoding="utf-8")
    except (OSError, configparser.Error):
        return {}
    if not read_files:
        return {}
    presets = {}
    for section in config.sections():
        presets[section] = dict(config.items(section))
    return presets


def _convert_config_scalar(value):
    text = value.strip()
    if not text:
        return text
    lowered = text.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    try:
        if lowered.startswith("0x"):
            return int(text, 16)
    except ValueError:
        pass
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _convert_config_value(value):
    text = value.strip()
    if not text:
        return text
    tokens = [tok for tok in text.replace(",", " ").split() if tok]
    if len(tokens) > 1:
        return [_convert_config_scalar(tok) for tok in tokens]
    return _convert_config_scalar(text)


def _resolve_preset_defaults(preset_values):
    defaults = {}
    for raw_key, raw_value in preset_values.items():
        dest_key = raw_key.replace('.', '_')
        defaults[dest_key] = _convert_config_value(raw_value)
    return defaults


def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = Path(__file__).resolve().parent

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))

    print('Backup Finished!')


def training(dataset, opt, pipe, pruning, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim,
        dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level,
        dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
    )
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False, logger=logger, resolution_scales=dataset.resolution_scales)
    gaussians.training_setup(opt, pruning)
    gaussians.set_coarse_interval(opt.coarse_iter, opt.coarse_factor)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt, pruning)

    mask_sampler = None
    mask_config = getattr(pruning, "mask", None) if pruning is not None else None
    mask_lambda_scale = 0.0
    mask_phase_breakpoints = (0.3, 0.7)
    mask_phase_weights = (1.0, 1.0, 1.0)
    mask_regularizer_enabled = False
    mask_regularizer_type = "l2"
    mask_global_interval = 0
    mask_diag_log_interval = 1000
    mask_diag_hitmap_enabled = False
    mask_diag_hitmap_interval = 1000
    mask_diag_top_k = 0
    mask_diag_trans_export = False
    mask_diag_trans_interval = 1000
    diagnostics_root = os.path.join(dataset.model_path, "diagnostics")
    mask_diag_hitmap_dir = None
    mask_diag_trans_dir = None
    mask_diag_inspection_dir = None
    mask_diag_inspection_views = []
    mask_diag_inspection_seed = 1337
    metrics_csv_path = os.path.join(diagnostics_root, "metrics.csv")
    metrics_csv_initialized = False

    if mask_config is not None:
        raw_lambda = getattr(mask_config, "lambda_m", 0.0)
        mask_lambda_scale = 0.0
        if isinstance(raw_lambda, str):
            if raw_lambda.lower() == "auto":
                mask_lambda_scale = 1.0
            else:
                try:
                    mask_lambda_scale = float(raw_lambda)
                except (TypeError, ValueError):
                    mask_lambda_scale = 0.0
        else:
            try:
                mask_lambda_scale = float(raw_lambda)
            except (TypeError, ValueError):
                mask_lambda_scale = 0.0
        scheduler_cfg = getattr(mask_config, "scheduler", None)
        if scheduler_cfg is not None:
            raw_breakpoints = getattr(scheduler_cfg, "phase_breakpoints", None)
            if raw_breakpoints:
                try:
                    values = sorted(min(max(float(bp), 0.0), 1.0) for bp in raw_breakpoints)
                except (TypeError, ValueError):
                    values = []
                if len(values) >= 2:
                    mask_phase_breakpoints = (values[0], values[1])
            raw_weights = getattr(scheduler_cfg, "phase_weights", None)
            if raw_weights is None:
                raw_weights = getattr(scheduler_cfg, "phase_scales", None)
            if raw_weights:
                try:
                    if isinstance(raw_weights, (list, tuple)):
                        weight_values = [float(val) for val in raw_weights]
                    else:
                        weight_values = [float(val) for val in str(raw_weights).replace(',', ' ').split()]
                    if len(weight_values) >= 3:
                        mask_phase_weights = (weight_values[0], weight_values[1], weight_values[2])
                except (TypeError, ValueError):
                    pass
        mask_lambda_scale = max(mask_lambda_scale, 0.0)
        mask_global_interval = int(getattr(mask_config, "global_interval", 0))
        raw_reg = getattr(mask_config, "regularizer", getattr(mask_config, "reg", "l2"))
        if isinstance(raw_reg, str):
            mask_regularizer_type = raw_reg.lower()
        else:
            mask_regularizer_type = str(raw_reg).lower()
        if mask_regularizer_type not in {"l1", "l2"}:
            mask_regularizer_type = "l2"

        diag_cfg = getattr(mask_config, "diagnostics", None)
        if diag_cfg is not None:
            try:
                mask_diag_log_interval = max(int(getattr(diag_cfg, "log_interval", mask_diag_log_interval)), 1)
            except (TypeError, ValueError):
                mask_diag_log_interval = 1000
            mask_diag_hitmap_enabled = bool(getattr(diag_cfg, "mask_hitmap_export", False))
            try:
                mask_diag_hitmap_interval = max(int(getattr(diag_cfg, "mask_hitmap_interval", mask_diag_log_interval)), 1)
            except (TypeError, ValueError):
                mask_diag_hitmap_interval = mask_diag_log_interval
            try:
                mask_diag_top_k = max(int(getattr(diag_cfg, "mask_hitmap_top_k", 0)), 0)
            except (TypeError, ValueError):
                mask_diag_top_k = 0
            hitmap_dir_name = getattr(diag_cfg, "mask_hitmap_dir", "mask_hitmap")
            mask_diag_hitmap_dir = os.path.join(diagnostics_root, hitmap_dir_name)

            mask_diag_trans_export = bool(getattr(diag_cfg, "transmittance_export", False))
            try:
                mask_diag_trans_interval = max(int(getattr(diag_cfg, "transmittance_interval", mask_diag_log_interval)), 1)
            except (TypeError, ValueError):
                mask_diag_trans_interval = mask_diag_log_interval
            trans_dir_name = getattr(diag_cfg, "transmittance_dir", "transmittance_heatmap")
            mask_diag_trans_dir = os.path.join(diagnostics_root, trans_dir_name)

            raw_views = getattr(diag_cfg, "inspection_views", [])
            if raw_views:
                mask_diag_inspection_views = [str(view) for view in raw_views]
            mask_diag_inspection_dir = os.path.join(diagnostics_root, getattr(diag_cfg, "inspection_dir", "inspection_views"))
            try:
                mask_diag_inspection_seed = int(getattr(diag_cfg, "inspection_rng_seed", mask_diag_inspection_seed))
            except (TypeError, ValueError):
                mask_diag_inspection_seed = 1337

    if mask_config is not None and getattr(mask_config, "enabled", False):
        mask_sampler = MaskSampler(mask_config, total_steps=opt.iterations)

    if (
        mask_config is not None
        and getattr(mask_config, "enabled", False)
        and mask_lambda_scale > 0.0
    ):
        mask_regularizer_enabled = True

    if mask_diag_hitmap_enabled or mask_diag_trans_export or mask_diag_inspection_views:
        os.makedirs(diagnostics_root, exist_ok=True)
        if os.path.isfile(metrics_csv_path) and os.path.getsize(metrics_csv_path) > 0:
            metrics_csv_initialized = True

    inspection_cameras = []
    if mask_diag_inspection_views:
        camera_lookup = {cam.image_name: cam for cam in scene.getTrainCameras()}
        for view_name in mask_diag_inspection_views:
            cam = camera_lookup.get(view_name)
            if cam is not None:
                inspection_cameras.append(cam)
            elif logger is not None:
                logger.warning("Inspection view %s not found in training cameras", view_name)

    def export_inspection_views(iteration_index: int, background_tensor: torch.Tensor) -> None:
        if not inspection_cameras or mask_diag_inspection_dir is None:
            return
        os.makedirs(mask_diag_inspection_dir, exist_ok=True)
        iteration_dir = os.path.join(mask_diag_inspection_dir, f"{iteration_index:06d}")
        os.makedirs(iteration_dir, exist_ok=True)

        sampler_enabled = mask_sampler is not None and getattr(mask_sampler, "enabled", False)
        rng_devices = []
        if background_tensor.is_cuda:
            rng_devices.append(background_tensor.device)

        with torch.no_grad():
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(mask_diag_inspection_seed)
                for cam in inspection_cameras:
                    gaussians.set_anchor_mask(cam.camera_center, iteration_index, cam.resolution_scale)
                    voxel_visible_mask = prefilter_voxel(cam, gaussians, pipe, background_tensor)
                    render_pkg = render(
                        cam,
                        gaussians,
                        pipe,
                        background_tensor,
                        visible_mask=voxel_visible_mask,
                        retain_grad=False,
                        mask_sampler=mask_sampler if sampler_enabled else None,
                        mask_step=iteration_index - 1,
                    )
                    render_image = torch.clamp(render_pkg["render"], 0.0, 1.0).detach().cpu()
                    torchvision.utils.save_image(render_image, os.path.join(iteration_dir, f"{cam.image_name}.png"))

    def mask_lambda_weight_for_step(step: int) -> float:
        if not mask_regularizer_enabled:
            return 0.0
        denom = max(opt.iterations - 1, 1)
        ratio = float(max(step, 0)) / float(denom)
        ratio = min(max(ratio, 0.0), 1.0)
        phase_a_end, phase_b_end = mask_phase_breakpoints
        phase_weight_a, phase_weight_b, phase_weight_c = mask_phase_weights
        if ratio < phase_a_end:
            base_lambda = 5e-4 * max(phase_weight_a, 0.0)
        elif ratio < phase_b_end:
            base_lambda = 8e-4 * max(phase_weight_b, 0.0)
        else:
            base_lambda = 1e-3 * max(phase_weight_c, 0.0)
        return mask_lambda_scale * base_lambda

    if logger is not None and hasattr(pruning, "mask"):
        logger.info("Mask pruning enabled: %s", getattr(pruning.mask, "enabled", False))

    if logger is not None and hasattr(pruning, "mask"):
        logger.info("Mask pruning enabled: %s", getattr(pruning.mask, "enabled", False))

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        densified_this_iter = False
        # network gui not available in octree-gs yet
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam,
                        gaussians,
                        pipe,
                        background,
                        scaling_modifer,
                        mask_sampler=mask_sampler,
                        mask_step=iteration - 1,
                    )["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if dataset.random_background:
            bg_color = [np.random.random(),np.random.random(),np.random.random()]
        elif dataset.white_background:
            bg_color = [1.0, 1.0, 1.0]
        else:
            bg_color = [0.0, 0.0, 0.0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        gaussians.set_anchor_mask(viewpoint_cam.camera_center, iteration, viewpoint_cam.resolution_scale)
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_diagnostics = None
        if mask_diag_hitmap_enabled and mask_diag_top_k > 0 and iteration % mask_diag_hitmap_interval == 0:
            render_diagnostics = {"mask_hitmap_enabled": True, "mask_hitmap_top_k": mask_diag_top_k}

        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            visible_mask=voxel_visible_mask,
            retain_grad=retain_grad,
            mask_sampler=mask_sampler,
            mask_step=iteration - 1,
            diagnostics=render_diagnostics,
        )

        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
        mask_hit_map = render_pkg.get("mask_topk_hit_map")
        final_trans_map = render_pkg.get("final_transmittance")

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        if scaling.shape[0] > 0:
            scaling_reg = scaling.prod(dim=1).mean()
        else:
            scaling_reg = torch.tensor(0.0, device="cuda")
        render_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg

        mask_keep_values = render_pkg.get("mask_keep_values")
        visibility_filter = render_pkg.get("visibility_filter")
        mask_loss = torch.zeros((), device=render_loss.device, dtype=render_loss.dtype)
        mask_mean = None
        mask_lambda_weight = 0.0
        if mask_keep_values is not None and mask_keep_values.numel() > 0:
            visible_keep_values = mask_keep_values
            if (
                visibility_filter is not None
                and visibility_filter.shape[0] == mask_keep_values.shape[0]
            ):
                visible_keep_values = mask_keep_values[visibility_filter]
            if visible_keep_values.numel() > 0:
                mask_mean = visible_keep_values.float().mean()
                if mask_regularizer_type == "l1":
                    mask_loss = mask_mean.abs()
                else:
                    mask_loss = mask_mean.square()
                mask_lambda_weight = mask_lambda_weight_for_step(iteration - 1)

        total_loss = render_loss + mask_loss * mask_lambda_weight

        total_loss.backward()

        iter_end.record()
        torch.cuda.synchronize()
        iter_time_ms = iter_start.elapsed_time(iter_end)

        mask_mean_value = mask_mean.detach() if mask_mean is not None else None
        mask_loss_value = mask_loss.detach()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            mask_metrics = None
            if mask_mean_value is not None:
                mean_keep = mask_mean_value.item()
                raw_mask_loss = mask_loss_value.item()
                weighted_mask_loss = raw_mask_loss * mask_lambda_weight
                mask_metrics = {
                    "mean_keep": mean_keep,
                    "raw_loss": raw_mask_loss,
                    "weighted_loss": weighted_mask_loss,
                    "lambda": mask_lambda_weight,
                    "regularizer": mask_regularizer_type,
                }

            # Log and save
            extra_metrics = None
            if iteration % mask_diag_log_interval == 0:
                with torch.no_grad():
                    psnr_value = psnr(image, gt_image).mean().item()
                    ssim_value = 1.0 - ssim_loss.detach().item()
                    lpips_value = lpips_fn(image, gt_image).mean().item()
                num_gaussians = gaussians.get_anchor.shape[0]
                mask_keep_scalar = float(mask_mean_value.item()) if mask_mean_value is not None else float("nan")
                render_fps = 1000.0 / max(iter_time_ms, 1e-6)
                extra_metrics = {
                    "num_gaussians": float(num_gaussians),
                    "mask_keep_mean": mask_keep_scalar,
                    "render_fps": render_fps,
                    "psnr": psnr_value,
                    "ssim": ssim_value,
                    "lpips": lpips_value,
                }

                if mask_diag_hitmap_enabled or mask_diag_trans_export or extra_metrics:
                    try:
                        os.makedirs(diagnostics_root, exist_ok=True)
                    except FileExistsError:
                        pass
                csv_row = {
                    "iteration": iteration,
                    "num_gaussians": extra_metrics["num_gaussians"],
                    "mask_keep_mean": extra_metrics["mask_keep_mean"],
                    "render_fps": extra_metrics["render_fps"],
                    "psnr": extra_metrics["psnr"],
                    "ssim": extra_metrics["ssim"],
                    "lpips": extra_metrics["lpips"],
                }
                with open(metrics_csv_path, "a", newline="") as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=["iteration", "num_gaussians", "mask_keep_mean", "render_fps", "psnr", "ssim", "lpips"])
                    if not metrics_csv_initialized or os.path.getsize(metrics_csv_path) == 0:
                        writer.writeheader()
                        metrics_csv_initialized = True
                    writer.writerow({k: ("" if v is None or (isinstance(v, float) and math.isnan(v)) else v) for k, v in csv_row.items()})
                if logger is not None:
                    logger.info(
                        "[ITER %d] Metrics: NG=%d, keep=%.4f, FPS=%.2f, PSNR=%.3f, SSIM=%.4f, LPIPS=%.4f",
                        iteration,
                        int(extra_metrics["num_gaussians"]),
                        extra_metrics["mask_keep_mean"],
                        extra_metrics["render_fps"],
                        extra_metrics["psnr"],
                        extra_metrics["ssim"],
                        extra_metrics["lpips"],
                    )

            training_report(
                tb_writer,
                dataset_name,
                iteration,
                Ll1,
                total_loss,
                l1_loss,
                iter_time_ms,
                testing_iterations,
                scene,
                render,
                (pipe, background),
                wandb,
                logger,
                mask_metrics=mask_metrics,
                extra_metrics=extra_metrics,
            )

            if (
                mask_diag_hitmap_enabled
                and mask_diag_hitmap_dir is not None
                and mask_diag_top_k > 0
                and iteration % mask_diag_hitmap_interval == 0
                and mask_hit_map is not None
                and mask_hit_map.numel() > 0
            ):
                os.makedirs(mask_diag_hitmap_dir, exist_ok=True)
                hitmap_path = os.path.join(mask_diag_hitmap_dir, f"{iteration:06d}.png")
                torchvision.utils.save_image(mask_hit_map.detach().cpu().unsqueeze(0), hitmap_path)

            if (
                mask_diag_trans_export
                and mask_diag_trans_dir is not None
                and iteration % mask_diag_trans_interval == 0
                and final_trans_map is not None
                and final_trans_map.numel() > 0
            ):
                os.makedirs(mask_diag_trans_dir, exist_ok=True)
                heatmap = (1.0 - final_trans_map.detach()).clamp_(0.0, 1.0).cpu()
                heatmap_path = os.path.join(mask_diag_trans_dir, f"{iteration:06d}.png")
                torchvision.utils.save_image(heatmap.unsqueeze(0), heatmap_path)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)

                # densification
                if opt.update_anchor and iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(
                        iteration=iteration,
                        check_interval=opt.update_interval,
                        success_threshold=opt.success_threshold,
                        grad_threshold=opt.densify_grad_threshold,
                        update_ratio=dataset.update_ratio,
                        extra_ratio=dataset.extra_ratio,
                        extra_up=dataset.extra_up,
                        min_opacity=opt.min_opacity
                    )
                    densified_this_iter = True
        if iteration == opt.update_until:
            del gaussians.opacity_accum
            del gaussians.offset_gradient_accum
            del gaussians.offset_denom
            torch.cuda.empty_cache()

        if mask_sampler is not None and getattr(mask_sampler, "enabled", False):
            prune_step_index = iteration - 1
            if densified_this_iter:
                gaussians.probabilistic_mask_prune(mask_sampler, prune_step_index, rng_offset=1)
            if mask_global_interval > 0 and iteration % mask_global_interval == 0:
                gaussians.probabilistic_mask_prune(mask_sampler, prune_step_index, rng_offset=2)

        if (
            inspection_cameras
            and mask_diag_inspection_dir is not None
            and mask_global_interval > 0
            and iteration % mask_global_interval == 0
        ):
            export_inspection_views(iteration, background)

        # Optimizer step
        if iteration < opt.iterations:
            if mask_sampler is not None and getattr(mask_sampler, "enabled", False):
                clip_params = []
                if gaussians._mask_logit_keep.grad is not None:
                    clip_params.append(gaussians._mask_logit_keep)
                if gaussians._mask_logit_drop.grad is not None:
                    clip_params.append(gaussians._mask_logit_drop)
                if clip_params:
                    clip_grad_norm_(clip_params, max_norm=1.0)
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
        if (iteration in checkpoint_iterations):
            logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
            torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(
    tb_writer,
    dataset_name,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
    wandb=None,
    logger=None,
    *,
    mask_metrics=None,
    extra_metrics=None,
):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)

        if mask_metrics is not None:
            mean_keep = mask_metrics.get("mean_keep")
            raw_loss = mask_metrics.get("raw_loss")
            weighted_loss = mask_metrics.get("weighted_loss")
            if mean_keep is not None:
                tb_writer.add_scalar(f'{dataset_name}/train_mask_keep_mean', mean_keep, iteration)
            if raw_loss is not None:
                tb_writer.add_scalar(f'{dataset_name}/train_mask_loss_raw', raw_loss, iteration)
            if weighted_loss is not None:
                tb_writer.add_scalar(f'{dataset_name}/train_mask_loss_weighted', weighted_loss, iteration)

        if extra_metrics is not None:
            for key, value in extra_metrics.items():
                if value is not None:
                    tb_writer.add_scalar(f'{dataset_name}/train_{key}', value, iteration)


    if wandb is not None:
        log_values = {"train_l1_loss": Ll1, "train_total_loss": loss}
        if mask_metrics is not None:
            if mask_metrics.get("mean_keep") is not None:
                log_values["mask_keep_mean"] = mask_metrics["mean_keep"]
            if mask_metrics.get("raw_loss") is not None:
                log_values["mask_loss_raw"] = mask_metrics["raw_loss"]
            if mask_metrics.get("weighted_loss") is not None:
                log_values["mask_loss_weighted"] = mask_metrics["weighted_loss"]
            if mask_metrics.get("lambda") is not None:
                log_values["mask_lambda"] = mask_metrics["lambda"]
        if extra_metrics is not None:
            for key, value in extra_metrics.items():
                if value is not None:
                    log_values[f"train_{key}"] = value
        wandb.log(log_values)

    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()

        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                                  {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0

                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    scene.gaussians.set_anchor_mask(viewpoint.camera_center, iteration, viewpoint.resolution_scale)
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()



                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))


                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    t_list = []
    visible_count_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize();t_start = time.time()

        gaussians.set_anchor_mask(view.camera_center, iteration, view.resolution_scale)
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()
        visible_count_list.append(visible_count)

        # gts
        gt = view.original_image[0:3, :, :]

        # error maps
        if gt.device != rendering.device:
            rendering = rendering.to(gt.device)
        errormap = (rendering - gt).abs()

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

    return t_list, visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=False, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim,
            dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level,
            dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=dataset.resolution_scales)
        gaussians.eval()

        if dataset.random_background:
            bg_color = [np.random.random(),np.random.random(),np.random.random()]
        elif dataset.white_background:
            bg_color = [1.0, 1.0, 1.0]
        else:
            bg_color = [0.0, 0.0, 0.0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })

    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, eval_name, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / eval_name

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())

        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)

            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)

        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)

def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    preset_name = _extract_config_preset(sys.argv[1:])
    preset_definitions = _load_mask_preset_definitions()
    available_presets = sorted(preset_definitions.keys())
    if preset_name and preset_name not in preset_definitions:
        raise SystemExit(
            f"Unknown config preset '{preset_name}'. Available presets: {', '.join(available_presets) if available_presets else 'none'}"
        )
    preset_defaults = _resolve_preset_defaults(preset_definitions.get(preset_name, {})) if preset_name else {}

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    pruning = PruningParams(parser)
    preset_arg_kwargs = {
        "type": str,
        "default": preset_name,
        "help": "Name of a configuration preset from configs/pruning_mask_presets.ini",
    }
    if available_presets:
        preset_arg_kwargs["choices"] = available_presets
    parser.add_argument('--config-preset', **preset_arg_kwargs)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    if preset_defaults:
        parser.set_defaults(**preset_defaults)
    args = parser.parse_args(sys.argv[1:])

    # enable logging
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)

    logger.info(f'args: {args}')

    if args.test_iterations[0] == -1:
        args.test_iterations = [i for i in range(10000, args.iterations + 1, 10000)]
    if len(args.test_iterations) == 0 or args.test_iterations[-1] != args.iterations:
        args.test_iterations.append(args.iterations)
    print(args.test_iterations)

    if args.save_iterations[0] == -1:
        args.save_iterations = [i for i in range(10000, args.iterations + 1, 10000)]
    if len(args.save_iterations) == 0 or args.save_iterations[-1] != args.iterations:
        args.save_iterations.append(args.iterations)
    print(args.save_iterations)

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    try:
        saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    except:
        logger.info(f'save code failed~')

    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]

    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Octree-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None

    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # record start time
    start_time = time.time()
    # training
    pruning_params = pruning.extract(args)

    training(lp.extract(args), op.extract(args), pp.extract(args), pruning_params, dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        training(lp.extract(args), op.extract(args), pp.extract(args), pruning_params, dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path)

    # All done
    logger.info(f"\nTraining complete. Total time: {time.time() - start_time:.2f} seconds.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    if args.eval:
        visible_count = render_sets(lp.extract(args), -1, pp.extract(args), skip_train=True, skip_test=False, wandb=wandb, logger=logger)
    else:
        visible_count = render_sets(lp.extract(args), -1, pp.extract(args), skip_train=False, skip_test=True, wandb=wandb, logger=logger)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    eval_name = 'test' if args.eval else 'train'
    evaluate(args.model_path, eval_name, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")