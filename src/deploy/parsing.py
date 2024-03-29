import _io
from io import BytesIO

import numpy as np
import streamlit as st
import plyfile as ply
from numpy.lib import recfunctions as rfn


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


def parse(file: st.runtime.uploaded_file_manager.UploadedFile) -> np.ndarray:
    """
    Parse point cloud.

    :param byte_obj: input object to be parsed.
    :type byte_obj: _io.BytesIO
    :returns: point cloud
    :rtype: np.ndarray
    """
    byte_object = BytesIO(file.getvalue())

    ext = file.__dict__["name"].split(".")[-1]
    if ext == "npz":
        return parse_npz(byte_object)
    elif ext == "npy":
        return parse_npy(byte_object)
    elif ext == "ply":
        return parse_ply(byte_object)
    elif ext == "xyz" or ext == "pts" or ext == "txt":
        return parse_xyz_pts_txt(byte_object, ext)
    elif ext == "csv":
        return parse_csv(byte_object)
