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

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from torch._C import _GLIBCXX_USE_CXX11_ABI
import os
os.path.dirname(os.path.abspath(__file__))

abi_flag = int(_GLIBCXX_USE_CXX11_ABI)

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={
                "cxx": [f"-D_GLIBCXX_USE_CXX11_ABI={abi_flag}"],
                "nvcc": [
                    f"-D_GLIBCXX_USE_CXX11_ABI={abi_flag}",
                    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")
                ],
            })
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
