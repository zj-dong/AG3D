from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
cuda_dir =  "training/deformers/fast_snarf/cuda"

setup(
    name='fuse',
    ext_modules=[
        CUDAExtension('fuse_cuda', 
        [f'{cuda_dir}/fuse_kernel/fuse_cuda.cpp',
        f'{cuda_dir}/fuse_kernel/fuse_cuda_kernel.cu']),
        CUDAExtension('filter_cuda', 
        [f'{cuda_dir}/filter/filter.cpp',
        f'{cuda_dir}/filter/filter_kernel.cu']),
        CUDAExtension('precompute_cuda', 
        [f'{cuda_dir}/precompute/precompute.cpp',
        f'{cuda_dir}/precompute/precompute_kernel.cu'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

