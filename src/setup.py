import setuptools

from Cython.Build import cythonize
import numpy as np

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="frmodel",
    version="0.1.0",
    author="Eve-ning",
    author_email="dev_evening@hotmail.com",
    description="The base package to support frmodel data processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Eve-ning/frmodel",
    packages=setuptools.find_packages(),
    ext_modules=cythonize("**/*.pyx", include_path=["src"]),
    include_dirs=np.get_include(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    setup_requires=["cython", "numpy"],
    install_requires=["numpy", "seaborn", "sklearn", "scikit-image", "tqdm", "plotly", "opencv-python"],
    python_requires='>=3.7',
    include_package_data=True,
)

package_data = { 'mypackage': ['mycythonmodule-filename.pyx']},
include_package_data = True

