import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_git_commit_number():
    return '1'

def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == '__main__':
    version = '0.1.0+%s' % get_git_commit_number()
    write_version_to_file(version, 'detzero_det/version.py')

    setup(
        name='detzero_det',
        version=version,
        description='The 3d object detection module of DetZero framework',
        install_requires=[
            'numpy',
            'torch',
            'numba',
            'tensorboardX',
            'easydict',
            'pyyaml'
        ],
        author='PJLab-ADLab',
        license='Apache License 2.0',
        packages=["detzero_det"],
        cmdclass={'build_ext': BuildExtension},
        ext_modules=[
            # make_cuda_ext(
            #     name='iou3d_nms_cuda',
            #     module='detzero_det.ops.iou3d_nms',
            #     sources=[
            #         'src/iou3d_cpu.cpp',
            #         'src/iou3d_nms_api.cpp',
            #         'src/iou3d_nms.cpp',
            #         'src/iou3d_nms_kernel.cu',
            #     ]
            # ),
            make_cuda_ext(
                    name='roiaware_pool3d_cuda',
                    module='detzero_det.ops.roiaware_pool3d',
                    sources=[
                        'src/roiaware_pool3d.cpp',
                        'src/roiaware_pool3d_kernel.cu',
                    ]
                ),
        ],
    )
