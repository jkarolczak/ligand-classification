import _io
from io import BytesIO

import numpy as np
import streamlit as st
from scipy.stats import norm

CCP4_TARGET_VOXEL_SIZE = 0.2


def parse_npz(byte_obj: _io.BytesIO) -> np.ndarray:
    """
    Parse point cloud saved as a compressed numpy array.

    :param byte_obj: input object to be parsed.
    :type byte_obj: _io.BytesIO
    :returns: point cloud
    :rtype: np.ndarray
    """
    blob_container = np.load(byte_obj, encoding="bytes")
    blob_name = blob_container.__dict__["files"][-1]
    return blob_container[blob_name]


def parse_npy(byte_obj: _io.BytesIO) -> np.ndarray:
    """
    Parse point cloud saved as a numpy array.

    :param byte_obj: input object to be parsed.
    :type byte_obj: _io.BytesIO
    :returns: point cloud
    :rtype: np.ndarray
    """
    blob = np.load(byte_obj, encoding="bytes")
    return blob


def _construct_blob(points: np.ndarray) -> np.ndarray:
    x, y, z = points[:, 0][:, np.newaxis], points[:, 1][:, np.newaxis], points[:, 2][:, np.newaxis]
    features = points[:, 3][:, np.newaxis]
    x_range = int(np.max(x)) + 1
    y_range = int(np.max(y)) + 1
    z_range = int(np.max(z)) + 1

    blob = np.zeros((x_range, y_range, z_range))
    blob[(x.astype(int), y.astype(int), z.astype(int))] = features
    return blob


def parse_ply(byte_obj: _io.BytesIO) -> np.ndarray:
    """
    Parse ply files
    ply file structure:
    x y z feature
    where x, y, z are coordinates and feature is a feature value at point with the given x, y, z coordinates

    :param byte_obj: input object to be parsed.
    :type byte_obj: _io.BytesIO
    :returns: point cloud
    :rtype: np.ndarray
    """
    import plyfile as ply
    from numpy.lib import recfunctions as rfn

    plydata = ply.PlyData.read(byte_obj)

    points = np.array(plydata.elements[0].data)
    points = rfn.structured_to_unstructured(points)

    return _construct_blob(points)


def parse_xyz_pts_txt(byte_obj: _io.BytesIO, ext: str) -> np.ndarray:
    """
    Parse xyz, pts and txt files
    - xyz and txt file structure: should contain only numerical data (without any header), each line should have
        the following format:
        x y z feature
        where x, y, z are coordinates and feature is a feature value at point with the given x, y, z coordinates;
        lines should be separated with \n character
    - pts file structure: in the first line there should be information about the number of points in the given
        point cloud, later the structure follows the xyz format structure

    :param byte_obj: input object to be parsed.
    :type byte_obj: _io.BytesIO
    :param ext: file extension
    :type ext: str
    :returns: point cloud
    :rtype: np.ndarray
    """
    if ext == 'pts':
        _ = byte_obj.readline()
    points = np.loadtxt(byte_obj, encoding="bytes")
    return _construct_blob(points)


def parse_csv(byte_obj: _io.BytesIO) -> np.ndarray:
    """
    Parse csv files
    - csv file structure: should contain only numerical data (without any header), each line should have
        the following format:
        x, y, z, feature
        where x, y, z are coordinates and feature is a feature value at point with the given x, y, z coordinates;
        lines should be separated with \n character

    :param byte_obj: input object to be parsed.
    :type byte_obj: _io.BytesIO
    :returns: point cloud
    :rtype: np.ndarray
    """
    try:
        points = np.loadtxt(byte_obj, delimiter=',', encoding="bytes")
    except ValueError:
        _ = byte_obj.readline()
        points = np.loadtxt(byte_obj, delimiter=',', encoding="bytes")
    return _construct_blob(points)


def resample_blob(blob: np.ndarray, target_voxel_size: float, unit_cell: np.ndarray, map_array: np.ndarray) -> np.ndarray:
    """
    Resamples a given blob to a target voxel size using the provided unit cell and map array.

    :param blob: The blob to be resampled.
    :type blob: numpy.ndarray
    :param target_voxel_size: The target voxel size (in Angstroms).
    :type target_voxel_size: float
    :param unit_cell: The unit cell dimensions (in Angstroms).
    :type unit_cell: tuple
    :param map_array: The map array.
    :type map_array: numpy.ndarray

    :return: The resampled blob.
    :rtype: numpy.ndarray
    """
    from scipy.ndimage import zoom

    blob = zoom(
        blob,
        [
            unit_cell[0] / target_voxel_size / map_array.shape[0],
            unit_cell[1] / target_voxel_size / map_array.shape[1],
            unit_cell[2] / target_voxel_size / map_array.shape[2],
        ],
        prefilter=False,
    )

    return blob


def _compute_density_threshold(map_array: np.ndarray) -> float:
    map_median = np.median(map_array)
    map_std = np.std(map_array)
    value_mask = (map_array < map_median - 0.5 * map_std) | (map_array > map_median + 0.5 * map_std)

    quantile_threshold = norm.cdf(2.8)
    density_threshold = np.quantile(map_array[value_mask], quantile_threshold)

    return density_threshold


def parse_ccp4(byte_obj: _io.BytesIO) -> np.ndarray:
    """
    Parse ccp4 and mrc files
    - ccp4 and mrc files structure: should meet the description on ccp-em website:
        https://www.ccpem.ac.uk/mrc_format/mrc2014.php

    :param byte_obj: input object to be parsed.
    :type byte_obj: _io.BytesIO
    :returns: point cloud
    :rtype: np.ndarray
    """
    from tempfile import NamedTemporaryFile
    import mrcfile

    with NamedTemporaryFile(delete=True) as tmp_file:
        tmp_file.write(byte_obj.read())
        tmp_file.flush()

        with mrcfile.open(tmp_file.name, mode="r") as file:
            map_array = np.asarray(file.data, dtype="float")

            order = (3 - file.header.maps, 3 - file.header.mapr, 3 - file.header.mapc)
            blob = np.moveaxis(a=map_array, source=(0, 1, 2), destination=order)

            unit_cell = np.zeros(6, dtype="float")
            cell = file.header.cella[["x", "y", "z"]]
            unit_cell[:3] = cell.astype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]).view(("<f4", 3))

            unit_cell[0], unit_cell[2] = unit_cell[2], unit_cell[0]
            unit_cell[3:] = 90.
            blob = resample_blob(blob, CCP4_TARGET_VOXEL_SIZE, unit_cell, map_array)

            density_threshold = _compute_density_threshold(map_array)
            blob[blob < density_threshold] = 0

    return blob


def _parse_bytes(byte_object, ext: str) -> np.ndarray:
    if ext == "npz":
        return parse_npz(byte_object)
    elif ext == "npy":
        return parse_npy(byte_object)
    elif ext == "ply":
        return parse_ply(byte_object)
    elif ext in ("xyz", "pts", "txt"):
        return parse_xyz_pts_txt(byte_object, ext)
    elif ext == "csv":
        return parse_csv(byte_object)
    elif ext in ("mrc", "ccp4", "map"):
        return parse_ccp4(byte_object)


def parse_streamlit(file: st.runtime.uploaded_file_manager.UploadedFile) -> np.ndarray:
    """
    Parse point cloud.

    :param byte_obj: input object to be parsed.
    :type byte_obj: _io.BytesIO
    :returns: point cloud
    :rtype: np.ndarray
    """
    byte_object = BytesIO(file.getvalue())
    ext = file.__dict__["name"].split(".")[-1]
    return _parse_bytes(byte_object, ext)


def parse_flask(file) -> np.ndarray:
    """
    Parse point cloud from a file uploaded via Flask request.

    :param file: File uploaded from a request.
    :type file: werkzeug.datastructures.FileStorage
    :returns: Parsed point cloud as a numpy array.
    :rtype: np.ndarray
    """
    byte_object = BytesIO(file.read())
    ext = file.filename.split(".")[-1].lower()
    return _parse_bytes(byte_object, ext)
