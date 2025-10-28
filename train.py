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
import numpy as np

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
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

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


def apply_spa_preset_options(args: Namespace, parser: ArgumentParser) -> str:
    valid_presets = {"off", "minimal", "full"}
    preset_raw = getattr(args, "spa_preset", "off")
    preset = str(preset_raw).lower()
    if preset not in valid_presets:
        parser.error(
            f"--spa_preset must be one of {sorted(valid_presets)} (got '{preset_raw}')"
        )

    setattr(args, "spa_preset", preset)
    if preset == "off":
        setattr(args, "spa_enable", False)
        setattr(args, "spa_keep_densify", False)
    elif preset == "minimal":
        setattr(args, "spa_enable", True)
        setattr(args, "spa_keep_densify", False)
    else:
        setattr(args, "spa_enable", True)
        setattr(args, "spa_keep_densify", True)

    return preset


def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim,
        dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level,
        dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
    )
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False, logger=logger, resolution_scales=dataset.resolution_scales)
    gaussians.training_setup(opt)
    gaussians.set_coarse_interval(opt.coarse_iter, opt.coarse_factor)

    log_fn = logger.info if logger is not None else print

    spa_preset = getattr(opt, "spa_preset", "off")
    spa_enabled = bool(getattr(opt, "spa_enable", False))
    spa_keep_densify = bool(getattr(opt, "spa_keep_densify", False))

    raw_kappa_total = getattr(opt, "kappa_total", None)
    raw_keep_ratio = getattr(opt, "keep_ratio", None)
    kappa_total = (
        int(raw_kappa_total)
        if raw_kappa_total is not None and raw_kappa_total > 0
        else None
    )
    keep_ratio = (
        float(raw_keep_ratio)
        if raw_keep_ratio is not None and raw_keep_ratio > 0
        else None
    )

    log_dir = getattr(dataset, "model_path", None)
    exp_name = None
    if log_dir is not None:
        try:
            exp_name = Path(log_dir).name
        except Exception:
            exp_name = None
    if exp_name is None:
        exp_name = dataset_name

    if kappa_total is not None:
        target_desc = f"kappa_total={kappa_total}"
    elif keep_ratio is not None:
        target_desc = f"keep_ratio={keep_ratio}"
    else:
        target_desc = "kappa_total=None"

    log_fn(
        f"[SPA] preset={spa_preset} enable={spa_enabled} keep_densify={spa_keep_densify} "
        f"start={opt.spa_start_iter} stop={opt.spa_stop_iter} "
        f"delta=[{opt.spa_delta_start}, {opt.spa_delta_end}] "
        f"interval=[{opt.spa_interval_warm}, {opt.spa_interval_stable}] {target_desc}"
    )

    spa_run_meta = {
        "preset": spa_preset,
        "enable": spa_enabled,
        "keep_densify": spa_keep_densify,
        "spa_start_iter": opt.spa_start_iter,
        "spa_stop_iter": opt.spa_stop_iter,
        "spa_delta_start": opt.spa_delta_start,
        "spa_delta_end": opt.spa_delta_end,
        "spa_interval_warm": opt.spa_interval_warm,
        "spa_interval_stable": opt.spa_interval_stable,
        "quota_update_interval": opt.quota_update_interval,
        "kappa_total": kappa_total,
        "keep_ratio": keep_ratio,
        "target_descriptor": target_desc,
        "alpha_occupancy": opt.alpha_occupancy,
        "anchor_m_min": opt.anchor_m_min,
        "age_grace_iters": opt.age_grace_iters,
        "new_level_bootstrap_ratio": opt.new_level_bootstrap_ratio,
        "hot_anchor_boost": opt.hot_anchor_boost,
        "hysteresis_M_out": opt.hysteresis_M_out,
        "hysteresis_M_in": opt.hysteresis_M_in,
    }

    if log_dir is not None:
        run_meta_path = Path(log_dir) / "run_meta.json"
        existing_meta = {}
        if run_meta_path.exists():
            try:
                with open(run_meta_path, "r", encoding="utf-8") as f:
                    existing_meta = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing_meta = {}
        existing_meta["spa"] = spa_run_meta
        try:
            with open(run_meta_path, "w", encoding="utf-8") as f:
                json.dump(existing_meta, f, indent=2, sort_keys=True)
            log_fn(f"[SPA] run_meta saved to {str(run_meta_path)}")
        except OSError as exc:
            log_fn(f"[SPA][WARN] failed to write run_meta.json: {exc}")

    spa_manager = None
    spa_started = False
    if spa_enabled:
        from spa_core import SpaManager

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        spa_manager = SpaManager(
            device,
            logger,
            spa_start_iter=opt.spa_start_iter,
            spa_stop_iter=opt.spa_stop_iter,
            spa_delta_start=opt.spa_delta_start,
            spa_delta_end=opt.spa_delta_end,
            spa_interval_warm=opt.spa_interval_warm,
            spa_interval_stable=opt.spa_interval_stable,
            quota_update_interval=opt.quota_update_interval,
            kappa_total=kappa_total,
            keep_ratio=keep_ratio,
            alpha_occupancy=opt.alpha_occupancy,
            anchor_m_min=opt.anchor_m_min,
            age_grace_iters=opt.age_grace_iters,
            hysteresis_M_out=opt.hysteresis_M_out,
            hysteresis_M_in=opt.hysteresis_M_in,
            new_level_bootstrap_ratio=opt.new_level_bootstrap_ratio,
            hot_anchor_boost=opt.hot_anchor_boost,
            log_dir=log_dir,
            exp_name=exp_name,
        )

        log_fn(
            f"[SPA] enable start={opt.spa_start_iter} stop={opt.spa_stop_iter} "
            f"delta=[{opt.spa_delta_start}, {opt.spa_delta_end}] "
            f"interval=[{opt.spa_interval_warm}, {opt.spa_interval_stable}] {target_desc}"
        )

    def maybe_start_spa():
        nonlocal spa_started
        if spa_manager is None or spa_started:
            return
        opacity = gaussians.get_opacity()
        if not isinstance(opacity, torch.Tensor) or opacity.numel() == 0:
            return
        level_tensor = None
        if hasattr(gaussians, "get_level"):
            level_tensor = gaussians.get_level()
            if isinstance(level_tensor, torch.Tensor):
                level_tensor = level_tensor.detach()
            else:
                level_tensor = None
        spa_manager.start(opacity.detach(), level=level_tensor, anchor_id=None)
        spa_started = True

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # network gui not available in octree-gs yet
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
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

        maybe_start_spa()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        gaussians.set_anchor_mask(viewpoint_cam.camera_center, iteration, viewpoint_cam.resolution_scale)
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)

        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        if scaling.shape[0] > 0:
            scaling_reg = scaling.prod(dim=1).mean()
        else:
            scaling_reg = torch.tensor(0.0, device="cuda")
        loss_base = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg
        loss = loss_base

        if (
            spa_manager is not None
            and spa_started
            and opt.spa_start_iter <= iteration <= opt.spa_stop_iter
        ):
            current_opacity = gaussians.get_opacity()
            loss = spa_manager.append_loss(loss, current_opacity, iteration)
            if iteration % 100 == 0:
                with torch.no_grad():
                    diff_norm = torch.norm(
                        current_opacity.detach() - spa_manager.z.to(current_opacity.device)
                    ).item()
                log_fn = logger.info if logger is not None else print
                log_fn(
                    f"[SPA] iter={iteration} loss_base={loss_base.item():.6f} "
                    f"loss_total={loss.item():.6f} #G={spa_manager.N} "
                    f"||a-z||2={diff_norm:.6f}"
                )

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)

                if spa_manager is not None and spa_started:
                    total_points = spa_manager.N
                    n_offsets = getattr(gaussians, "n_offsets", None)
                    flat_count = offset_selection_mask.numel()
                    if (
                        isinstance(total_points, int)
                        and total_points > 0
                        and isinstance(n_offsets, int)
                        and n_offsets > 0
                        and flat_count == total_points * n_offsets
                    ):
                        selected_indices = torch.nonzero(offset_selection_mask, as_tuple=False).squeeze(-1)
                        full_visibility = torch.zeros(flat_count, dtype=torch.bool, device=offset_selection_mask.device)
                        if selected_indices.numel() == visibility_filter.numel():
                            full_visibility[selected_indices] = visibility_filter
                        anchor_visibility = full_visibility.view(total_points, n_offsets).any(dim=1)
                        spa_manager.ingest_visibility(anchor_visibility)

                        grad_full = torch.zeros(flat_count, dtype=torch.float32, device=offset_selection_mask.device)
                        if (
                            selected_indices.numel() == visibility_filter.numel()
                            and viewspace_point_tensor.grad is not None
                            and visibility_filter.any()
                        ):
                            grad_values = torch.norm(
                                viewspace_point_tensor.grad[visibility_filter, :2],
                                dim=-1,
                            )
                            grad_full[selected_indices[visibility_filter]] = grad_values
                        grad_anchor = grad_full.view(total_points, n_offsets).mean(dim=1)
                        grad_anchor = torch.nan_to_num(grad_anchor, nan=0.0, posinf=0.0, neginf=0.0)
                        spa_manager.ingest_grad(grad_anchor)

                        if iteration % 500 == 0:
                            vis_mean = float(spa_manager.vis_hit_ema.mean().item())
                            grad_mean = float(spa_manager.grad_ema.mean().item())
                            log_fn = logger.info if logger is not None else print
                            log_fn(
                                f"[SPA] stats iter={iteration} vis_hit_ema={vis_mean:.6f} grad_ema={grad_mean:.6f}"
                            )

                # densification
                if opt.update_anchor and iteration > opt.update_from and iteration % opt.update_interval == 0:
                    spa_keep_densify = getattr(opt, "spa_keep_densify", False)
                    skip_densify = (
                        spa_manager is not None
                        and spa_started
                        and getattr(spa_manager, "enabled", False)
                        and opt.spa_start_iter <= iteration <= opt.spa_stop_iter
                        and not spa_keep_densify
                    )
                    if not skip_densify:
                        anchor_accessor = getattr(gaussians, "get_anchor")
                        anchor_tensor_before = anchor_accessor() if callable(anchor_accessor) else anchor_accessor
                        N_before = int(anchor_tensor_before.shape[0])
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
                        anchor_accessor_after = getattr(gaussians, "get_anchor")
                        anchor_tensor_after = anchor_accessor_after() if callable(anchor_accessor_after) else anchor_accessor_after
                        N_after = int(anchor_tensor_after.shape[0])
                        if (
                            spa_manager is not None
                            and spa_started
                            and spa_keep_densify
                        ):
                            current_opacity = gaussians.get_opacity()
                            resized = False
                            spa_N_before = spa_manager.N
                            if isinstance(current_opacity, torch.Tensor):
                                if N_after != spa_N_before:
                                    spa_manager.resize_on_density(N_after, current_opacity.detach())
                                    resized = N_after > spa_N_before
                                level_tensor_attr = getattr(gaussians, "get_level", None)
                                level_tensor = None
                                if level_tensor_attr is not None:
                                    level_value = level_tensor_attr() if callable(level_tensor_attr) else level_tensor_attr
                                    if isinstance(level_value, torch.Tensor):
                                        level_tensor = level_value.detach()
                                anchor_id_attr = getattr(gaussians, "get_anchor_id", None)
                                anchor_id_tensor = None
                                if anchor_id_attr is not None:
                                    anchor_id_value = anchor_id_attr() if callable(anchor_id_attr) else anchor_id_attr
                                    if isinstance(anchor_id_value, torch.Tensor):
                                        anchor_id_tensor = anchor_id_value.detach()
                                if level_tensor is not None or anchor_id_tensor is not None:
                                    spa_manager.set_meta(level_tensor, anchor_id_tensor)
                            log_fn = logger.info if logger is not None else print
                            log_fn(
                                f"[SPA] densify iter={iteration} N_before={N_before} N_after={N_after} resized={resized}"
                            )
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (
                spa_manager is not None
                and spa_started
                and getattr(spa_manager, "enabled", True)
                and opt.spa_start_iter <= iteration <= opt.spa_stop_iter
            ):
                interval_val = spa_manager.interval(iteration)
                if interval_val > 0 and iteration % interval_val == 0:
                    current_opacity = gaussians.get_opacity()
                    if isinstance(current_opacity, torch.Tensor):
                        spa_manager.step_prox(current_opacity.detach(), iteration)
            if (
                spa_manager is not None
                and spa_started
                and iteration == opt.spa_stop_iter
            ):
                prune_mask = spa_manager.build_prune_mask()
                removed = int(prune_mask.sum().item())
                if removed > 0:
                    anchor_tensor = gaussians.get_anchor()
                    prune_mask_for_gaussians = prune_mask.to(
                        device=anchor_tensor.device, dtype=torch.bool
                    )
                    gaussians.prune_points(prune_mask_for_gaussians)
                spa_manager.apply_prune_mask(prune_mask)
                remain_total = int(spa_manager.N)
                per_level_remain = spa_manager.remaining_per_level()
                if not per_level_remain:
                    per_level_remain = {0: remain_total}
                per_level_items = ", ".join(
                    f"L{lvl}:{count}" for lvl, count in sorted(per_level_remain.items())
                )
                log_fn = logger.info if logger is not None else print
                log_fn(
                    f"[SPA] PRUNE: removed={removed} remain={remain_total} "
                    f"per-layer remain={{ {per_level_items} }} "
                    f"final_kappa_total={spa_manager.current_kappa_total()}"
                )
                pruned_checkpoint_path = os.path.join(scene.model_path, "chkpnt_pruned.pth")
                torch.save((gaussians.capture(), iteration), pruned_checkpoint_path)
                log_fn(f"[SPA] pruned checkpoint saved to {pruned_checkpoint_path}")
                snapshot_paths = spa_manager.snapshot_logs(tag="pruned")
                if snapshot_paths:
                    log_fn(
                        "[SPA] metrics snapshot saved: "
                        + ", ".join(str(path) for path in snapshot_paths)
                    )
                else:
                    log_fn("[SPA] metrics snapshot skipped (no log files)")
                spa_manager.enabled = False
                log_fn("[SPA] finetune phase: proximal updates paused")
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

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })

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
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
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
    args = parser.parse_args(sys.argv[1:])
    apply_spa_preset_options(args, parser)

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
    training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path)

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