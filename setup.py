# Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

# This file is part of the implementation as described in the NIPS 2018 paper:
# Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
# Please see the file LICENSE.txt for the license governing this code.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='n3net_matmul',
    package_dir={"": "lib"},
    packages=["n3net"],
    ext_modules=[
        CUDAExtension('n3net_matmul_cuda', [
            'lib/csrc/matmul.cpp',
            'lib/csrc/matmul1_kernel.cu',
            'lib/csrc/matmul1_bwd_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
