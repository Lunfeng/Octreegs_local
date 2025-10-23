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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    masks,
    mask_keep_probabilities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        masks,
        mask_keep_probabilities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        masks,
        mask_keep_probabilities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        if hasattr(ctx, "set_materialize_grads"):
            ctx.set_materialize_grads(False)

        # Ordered inputs to the CUDA kernel (_C.rasterize_gaussians) with their autograd status:
        #   0  bg (non-differentiable)
        #   1  means3D (differentiable)
        #   2  colors_precomp (differentiable)
        #   3  opacities (differentiable)
        #   4  masks (logical, detach)
        #   5  mask_keep_probabilities (logical, detach)
        #   6  scales (differentiable)
        #   7  rotations (differentiable)
        #   8  scale_modifier (non-differentiable scalar)
        #   9  cov3Ds_precomp (differentiable)
        #   10 viewmatrix (non-differentiable)
        #   11 projmatrix (non-differentiable)
        #   12 tanfovx (non-differentiable scalar)
        #   13 tanfovy (non-differentiable scalar)
        #   14 image_height (non-differentiable int)
        #   15 image_width (non-differentiable int)
        #   16 sh (differentiable)
        #   17 sh_degree (non-differentiable int)
        #   18 campos (non-differentiable)
        #   19 prefiltered (non-differentiable bool)
        #   20 debug (non-differentiable bool)
        #   21 diagnostics_enable_mask_hitmap (non-differentiable bool)
        #   22 diagnostics_mask_top_k (non-differentiable int)
        # Non-differentiable logical inputs such as masks and mask_keep_probabilities must be detached.
        if masks is None:
            masks = torch.empty(0, dtype=torch.uint8, device=means3D.device)
        else:
            masks = masks.detach()
        if mask_keep_probabilities is None:
            mask_keep_probabilities = torch.empty(0, dtype=means3D.dtype, device=means3D.device)
        else:
            mask_keep_probabilities = mask_keep_probabilities.detach()

        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            masks,
            mask_keep_probabilities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.diagnostics_enable_mask_hitmap,
            int(raster_settings.diagnostics_mask_top_k),
        )

        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                num_rendered, color, radii, final_T, mask_hit_map, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, final_T, mask_hit_map, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, final_T, mask_hit_map

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_radii, grad_out_final_T, grad_out_mask_hit):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_list = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_list = _C.rasterize_gaussians_backward(*args)

        grad_list = tuple(grad_list)
        expected_inputs = 11
        if len(grad_list) != expected_inputs:
            raise RuntimeError(
                f"Backward must return {expected_inputs} grads, got {len(grad_list)}"
            )

        def optional_grad(tensor):
            return tensor if tensor is not None else None

        (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_masks,
            grad_mask_keep_probabilities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_raster_settings,
        ) = grad_list

        grads = (
            optional_grad(grad_means3D),
            optional_grad(grad_means2D),
            optional_grad(grad_sh),
            optional_grad(grad_colors_precomp),
            optional_grad(grad_opacities),
            optional_grad(grad_masks),
            optional_grad(grad_mask_keep_probabilities),
            optional_grad(grad_scales),
            optional_grad(grad_rotations),
            optional_grad(grad_cov3Ds_precomp),
            optional_grad(grad_raster_settings),
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    diagnostics_enable_mask_hitmap: bool = False
    diagnostics_mask_top_k: int = 0

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, masks = None, mask_keep_probabilities = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if masks is None:
            masks = torch.empty(0, dtype=torch.uint8, device=means3D.device)

        if mask_keep_probabilities is None:
            mask_keep_probabilities = torch.empty(0, dtype=means3D.dtype, device=means3D.device)

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            masks,
            mask_keep_probabilities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )

    def visible_filter(self, means3D, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        with torch.no_grad():
            radii = _C.rasterize_aussians_filter(means3D,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.prefiltered,
            raster_settings.debug)
        return  radii
    
    


