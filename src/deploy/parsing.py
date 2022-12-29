import _io
from io import BytesIO

import numpy as np
import streamlit as st
import plyfile as ply


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


def parse_ply(byte_obj: _io.BytesIO) -> np.ndarray:
    """
    ply file structure:
     x y z feature
     where x, y, z are coordinates and feature is a feature value at point with the given x, y, z coordinates
    :param byte_obj: input object to be parsed.
    :type byte_obj: _io.BytesIO
    :returns: point cloud
    :rtype: np.ndarray

    """
    plydata = ply.PlyData.read(byte_obj)

    properties_names = [str(p.name) for p in plydata.elements[0].properties]

    x = np.array(plydata.elements[0].data[properties_names[0]])[:, np.newaxis]
    y = np.array(plydata.elements[0].data[properties_names[1]])[:, np.newaxis]
    z = np.array(plydata.elements[0].data[properties_names[2]])[:, np.newaxis]
    features = np.array(plydata.elements[0].data[properties_names[-1]])[:, np.newaxis]

    x_range = int(np.max(x)) + 1
    y_range = int(np.max(y)) + 1
    z_range = int(np.max(z)) + 1

    blob = np.zeros((x_range, y_range, z_range))
    blob[(x.astype(int), y.astype(int), z.astype(int))] = features

    return blob


def parse_pcd(byte_obj: _io.BytesIO) -> np.ndarray:
    pass


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
    elif ext == 'ply':
        return parse_ply(byte_object)
    elif ext == 'pcd':
        return parse_pcd(byte_object)
