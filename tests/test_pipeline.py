import os

import numpy as np
import pytest

from pipeline.transforms import *


def blobs():
    files = os.listdir("../static/tests-data")
    _blobs = []
    for idx, f_name in enumerate(files):
        input_path = os.path.join("../static/tests-data", f_name)
        _blob = np.load(input_path)
        _blobs.append(_blob["blob"])
    return _blobs


@pytest.mark.parametrize("config", [
    {'max_blob_size': 2000, 'method': 'basic'},
    {'max_blob_size': 2000, 'method': 'average'},
    {'max_blob_size': 2000, 'method': 'max'},
    {'max_blob_size': 500, 'method': 'basic'},
    {'max_blob_size': 500, 'method': 'average'},
    {'max_blob_size': 500, 'method': 'max'}])
def test_uniform_selection_transform(config):
    transforms = UniformSelectionTransform(config)
    for blob in blobs():
        transformed = transforms.preprocess(blob)
        assert transformed.shape == blob.shape
        assert np.sum(transformed > 0) <= config['max_blob_size']


@pytest.mark.parametrize("config", [{"neighbourhood": 6}, {"neighbourhood": 22}, {"neighbourhood": 26}])
def test_blob_surface_transform(config):
    transforms = BlobSurfaceTransform(config)
    for blob in blobs():
        transformed = transforms.preprocess(blob)
        assert transformed.shape == blob.shape
        assert np.sum(transformed > 0) <= np.sum(transformed > 0)


@pytest.mark.parametrize("config", [{"max_blob_size": 200}, {"max_blob_size": 2000}, {"max_blob_size": 10000}])
def test_random_selection(config):
    transforms = RandomSelectionTransform(config)
    for blob in blobs():
        transformed = transforms.preprocess(blob)
        assert transformed.shape == blob.shape
        assert np.sum(transformed > 0) <= np.sum(transformed > 0)


@pytest.mark.parametrize("config", [{"max_blob_size": 200}, {"max_blob_size": 1000}, {"max_blob_size": 2000}])
def test_clustering(config):
    transforms = ClusteringTransform(config)
    for blob in blobs():
        transformed = transforms.preprocess(blob)
        assert transformed.shape == blob.shape
        assert np.sum(transformed > 0) <= np.sum(transformed > 0)
