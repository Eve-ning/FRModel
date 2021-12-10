Forest Recovery Modelling
=========================

This is a Python Package to help those that are under the URECA (NTU) Forest Recovery Modelling Team

.. toctree::
   :maxdepth: 2

   Information <info>

   Base <frmodel/base>

---------------------------
Installation Method 1: Fork
---------------------------

While not recommended for enterprise packages, I rarely update `pip` unless it's required.

To get the latest updates, just fork this repository.

Note that you will need to build the **Cython** code, the `c_setup.bat` can be found in the root directory.

----------------------------------
Installation Method 2: Conda + Pip
----------------------------------

**Environment: Python 3.8 Anaconda**

Only Anaconda works because of `gdal`. It will not work with `pip venv` unless you can install `gdal` on that.

However, the package is only on **PyPI**, you can still do `pip install frmodel` and it will still work on Anaconda
environments.

When installing, you should receive these new packages, manually install them if missing.

`gdal` **needs** to be installed with `conda install gdal -c conda-forge`.

.. code-block::

    numpy
    seaborn
    sklearn
    skimage
    tqdm
    plotly
    opencv
    gdal *



