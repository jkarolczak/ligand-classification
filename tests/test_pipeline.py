import os

import numpy as np
import pytest

from pipeline.transforms import BlobSurfaceTransform


def blobs():
    files = os.listdir("../static/tests-data")
    _blobs = []
    for idx, f_name in enumerate(files):
        input_path = os.path.join("../static/tests-data", f_name)
        _blob = np.load(input_path)
        _blobs.append(_blob["blob"])
    return _blobs


@pytest.mark.parametrize("config", [{"spacing": [0.0, 0.0, 0.0], "method": "lorensen"}])
def test_blob_surface_transform(config):
    transforms = BlobSurfaceTransform(config)
    for blob in blobs():
        transformed = transforms.preprocess(blob)
        assert transformed.shape == blob.shape
        assert np.sum(transformed > 0) <= np.sum(transformed > 0)
