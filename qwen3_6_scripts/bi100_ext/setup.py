from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent

setup(
    name="bi100-gdn-recurrent",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="bi100_gdn_recurrent",
            sources=[
                str(ROOT / "bindings.cpp"),
                str(ROOT / "gdn_recurrent.cu"),
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
