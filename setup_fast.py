from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules=[ Extension("program",
              ["program.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math"])]

setup(
  name = "program",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules,
  include_dirs=[numpy.get_include()])
