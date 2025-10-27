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

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    """Compute PSNR for batched or single images."""
    if img1.shape == img2.shape and img1.ndim == 3:
        mse = torch.mean((img1 - img2) ** 2)
        mse = torch.clamp(mse, min=1e-12)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

    flat1 = img1.reshape(img1.shape[0], -1)
    flat2 = img2.reshape(img2.shape[0], -1)
    mse = torch.mean((flat1 - flat2) ** 2, dim=1)
    mse = torch.clamp(mse, min=1e-12)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
