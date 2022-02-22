import os

from tqdm import tqdm
import numpy as np
import pytest

from frmodel.base.D2 import Frame2D

DATA_PATH = "C:/Users/johnc/OneDrive - Nanyang Technological University/NTU/URECA/Data/"


@pytest.mark.parametrize(
    'search_path', ['v2_10May2021', 'v2_18Dec2020']
)
def test_report(search_path):
    """ Generates the reports for glcms

    This will generate the report for both the histogram and image

    :param search_path: Path to search in, can be in subdirs
    :return:
    """

    for root, dirs, files in tqdm(os.walk(f"{search_path}/")):
        for name in files:
            # Only GLCMs
            if not name.endswith('.npz'): continue
            path = os.path.join(root, name)
            print("Generating Report on ", path)
            path_no_ext = path[:-4]  # This removes the .npz extension
            f = Frame2D.load(f"{path}")
            assert not np.isnan(f.data.data).any(), "NaN found in GLCM, aborting"
            f.plot().hist().savefig(f"{path_no_ext}-hist.jpg")
            f.plot().image().savefig(f"{path_no_ext}-image.jpg")
