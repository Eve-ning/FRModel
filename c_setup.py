from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='frmodel',
    ext_modules=cythonize("src/**/*.pyx", build_dir="cython-build",
                          annotate=True),
    package_dir={'FRModel': ''},
    include_dirs=[numpy.get_include()]
)
