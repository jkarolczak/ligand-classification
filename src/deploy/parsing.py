import _io
from io import BytesIO

import numpy as np
import streamlit as st


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
