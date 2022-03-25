import os

import numpy as np
import pytest

from pipeline.transforms import UniformSelectionTransform


def blobs():
    files = os.listdir("../static/tests-data")
    _blobs = []
    for idx, f_name in enumerate(files):
        input_path = os.path.join("../static/tests-data", f_name)
        _blob = np.load(input_path)
        _blobs.append(_blob["blob"])
    return _blobs


@pytest.mark.parametrize("config", [
    {'max_voxel': 2000, 'method': 'basic'},
    {'max_voxel': 2000, 'method': 'average'},
    {'max_voxel': 2000, 'method': 'max'},
    {'max_voxel': 500, 'method': 'basic'},
    {'max_voxel': 500, 'method': 'average'},
    {'max_voxel': 500, 'method': 'max'}])
def test_uniform_selection_transform(config):
    transforms = UniformSelectionTransform(config)
    for blob in blobs():
        transformed = transforms.preprocess(blob)
        assert transformed.shape == blob.shape
        assert np.sum(transformed > 0) <= config['max_voxel']