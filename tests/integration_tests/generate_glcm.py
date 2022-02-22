import os

import numpy as np
import pytest

from frmodel.data.load import load_spec

DATA_PATH = "C:/Users/johnc/OneDrive - Nanyang Technological University/NTU/URECA/Data/"

@pytest.mark.parametrize(
    'version',
    ['v2']
)
@pytest.mark.parametrize(
    'path, output_dir_name',
    [
        ('imgs/spec/chestnut/10May2021/90deg43m85pct255deg/map/', '10May2021'),
        ('imgs/spec/chestnut/18Dec2020/', '18Dec2020'),
    ]
)
@pytest.mark.parametrize(
    'bins',
    [128]
)
def test_generate_glcm(version, path, output_dir_name, bins):
    frame, trees = load_spec(
        f"{DATA_PATH}{path}",
        ignore_broadband=True
    )
    output_dir = f"{version}_{output_dir_name}/{int(np.log2(bins))}/"
    os.makedirs(output_dir, exist_ok=True)
    n = None
    i = 0
    for tree in trees:
        output_path = output_dir + tree.name + str(i) + ".npz"
        print(output_path)
        if tree.name == n:
            i += 1

        if not os.path.exists(output_path):
            tree.frame.get_glcm(bins=bins)
            tree.frame.save(output_dir + tree.name + str(i) + ".npz")

        n = tree.name
