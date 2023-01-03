import os

import pytest

from pipeline.transforms import *


def blobs(path: str = "../static/tests-data"):
    files = os.listdir(path)
    _blobs = []
    for idx, f_name in enumerate(files):
        input_path = os.path.join(path, f_name)
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


@pytest.mark.parametrize("config", [{"neighbourhood": 6}, {"neighbourhood": 18}, {"neighbourhood": 26}])
def test_blob_surface_transform(config):
    transforms = BlobSurfaceTransform(config)
    for blob in blobs():
        transformed = transforms.preprocess(blob)
        assert transformed.shape == blob.shape
        assert np.sum(transformed > 0) <= np.sum(blob > 0)


@pytest.mark.parametrize("config", [{"max_blob_size": 200}, {"max_blob_size": 2000}, {"max_blob_size": 10000}])
def test_random_selection(config):
    transforms = RandomSelectionTransform(config)
    for blob in blobs():
        transformed = transforms.preprocess(blob)
        assert transformed.shape == blob.shape
        if np.sum(blob > 0) > config["max_blob_size"]:
            assert np.sum(transformed > 0) == config["max_blob_size"]
        else:
            assert np.sum(transformed > 0) == np.sum(blob > 0)


@pytest.mark.parametrize("config", [
    {"max_blob_size": 200, "n_init": 1, "max_iter": 5},
    {"max_blob_size": 1000, "n_init": 1, "max_iter": 5},
    {"max_blob_size": 2000, "n_init": 1, "max_iter": 5}])
def test_clustering(config):
    transforms = ClusteringTransform(config)
    for blob in blobs():
        transformed = transforms.preprocess(blob)
        assert transformed.shape == blob.shape
        if np.sum(blob > 0) > config["max_blob_size"]:
            assert np.sum(transformed > 0) == config["max_blob_size"]
        else:
            assert np.sum(transformed > 0) == np.sum(blob > 0)


def padding(array, xx, yy, zz):
    """
    Based on https://stackoverflow.com/questions/59241216/padding-numpy-arrays-to-a-specific-size
    pad blobs to have the same shape for comparison
    :param array: numpy array
    :param xx: desired height
    :param yy: desired width
    :param zz: desired depth
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]
    d = array.shape[2]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    c = (zz - d) // 2
    cc = zz - c - d

    return np.pad(array, pad_width=((a, aa), (b, bb), (c, cc)), mode='constant')


def test_pca_transform():
    transforms = PCATransform()
    a, b = blobs("../static/pca-tests-blobs")

    a_pad = padding(a, 100, 100, 100)
    b_pad = padding(b, 100, 100, 100)
    before = np.sum(a_pad - b_pad)

    a_rotated = transforms.preprocess(a)
    b_rotated = transforms.preprocess(b)

    a_rot_pad = padding(a_rotated, 100, 100, 100)
    b_rot_pad = padding(b_rotated, 100, 100, 100)
    rotated = np.sum(a_rot_pad - b_rot_pad)
    assert np.abs(before) > np.abs(rotated)
