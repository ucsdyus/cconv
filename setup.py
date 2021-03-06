from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import glob

include_dir = os.path.dirname(os.path.abspath(__file__))
cc_sources = glob.glob(os.path.join(include_dir, "", "*.cc"))
cu_sources = glob.glob(os.path.join(include_dir, "", "*.cu"))

print("cc sources", cc_sources)
print("cu sources", cu_sources)

setup(name="FastPatch",
      ext_modules=[CUDAExtension(
          "fastpatch_impl", sources=cc_sources + cu_sources,
          extra_compile_args={
           #    "cxx": ["/O2", "/w", "/std:c++11"],
              "cxx": ["-O3", "-w", "-std=c++11"],
              "nvcc": ["-O3", "--ptxas-options=-v", "-w"]})],
      cmdclass={"build_ext": BuildExtension})
