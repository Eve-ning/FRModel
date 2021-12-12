import os
from distutils.extension import Extension

from setuptools import setup
from Cython.Build import cythonize

import numpy
# os.environ["CC"] = "clang++"

a = cythonize("**/*.pyx",
              include_path=["src"],
              build_dir="cython-build",
              annotate=True,
              language_level="3")


setup(
    name='frmodel',
    ext_modules=a,
    package_dir={'src': ''},
    include_dirs=[numpy.get_include()]
)
