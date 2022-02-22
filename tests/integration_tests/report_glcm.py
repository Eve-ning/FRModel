import os

import numpy as np
import pytest
from matplotlib import pyplot as plt

from frmodel.base.D2 import Frame2D
from frmodel.data.load import load_spec

DATA_PATH = "C:/Users/johnc/OneDrive - Nanyang Technological University/NTU/URECA/Data/"


@pytest.mark.parametrize(
    'search_path', ['v2_10May2021', 'v2_18Dec2020']
)
def test_report(search_path):
    """ Generates the reports for glcms

    :param search_path: Path to search in, can be in subdirs
    :return:
    """

    # ar = Frame2D.load(f"{path}/4")

    for root, dirs, files in os.walk(f"{search_path}/"):
        for name in files:
            # Only GLCMs
            if not name.endswith('.npz'): continue

            path = os.path.join(root, name)
            f = Frame2D.load(f"{path}")
            f.plot().hist().savefig("")
            print(os.path.join(root, name))

    # nans = np.isnan(ar.data.data).any(axis=0).any(axis=0)
    # plt.imshow(ar.data.data[..., 0])
    # plt.show()
    # ar.plot().hist().savefig("out.png")
