from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='frmodel',
    ext_modules=cythonize("**/*.pyx",
                          include_path=["src"],
                          build_dir="cython-build",
                          annotate=True),
    package_dir={'src': ''},
    include_dirs=[numpy.get_include()]
)
