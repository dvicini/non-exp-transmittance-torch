from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nonexp_cuda',
    py_modules=['nonexp'],
    ext_modules=[
        CUDAExtension('_nonexp_cuda', [
            'non-exp-transmittance.cpp',
            'non-exp-transmittance-kernels.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    })
