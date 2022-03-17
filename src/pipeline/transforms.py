import os
from abc import ABC, abstractmethod
from typing import Dict, Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy.ndimage import zoom
import math
from skimage.measure import marching_cubes
from plotting import plot_interactive_trisurf

"""
while writing our functions, we can specify the expected type of arguments and return. This is especially useful
for people browsing and reviewing our code. Almighty Python language offers some useful helpers:

using:

from typing import List, Tuple, Union

we can be more specific, e.g.
    def adam(grzenda: List[str]) -> Tuple[int]:

Union[str, List[str]] -> either string or a list of strings
"""


class Transform(ABC):
    """
    Abstract class for preprocessing transformations
    """

    def __init__(self, config: Union[Dict, None] = None, **kwargs) -> None:
        self.__dict__.update(config)

    @abstractmethod
    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        """
        Abstract method for applying preprocessing methods to a blob
        :param blob: blob to be processed
        """
        pass


class BlobSurfaceTransform(Transform):
    """
    A class that limit voxels in the blob by removing all voxels that don't create surface of the blob. It removes
    voxels that don't have any 0 in their neighbourhood. This class extends `Transformer` class.
    """

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        blob = marching_cubes(blob, spacing=self.spacing, method=self.method)
        coords = blob[0].astype(int)
        shape = coords.max(-2)
        values = blob[3]
        blob = np.zeros(shape)
        np.put(blob, coords, values)
        return blob


class UniformSelectionTransform(Transform):
    """
    A class that limits the number of voxels in the blob by either selecting  This class extends 'Transformer' class.

    :param config: configuration dictionary with integer 'max_voxel' and str 'method' entries
    """

    def __init__(self, config: Union[Dict, None] = None, **kwargs) -> None:
        super().__init__(config, **kwargs)
        if self.__dict__.get('max_voxel') is None:
            raise ValueError("{} requires 'max_voxel' key in config dictionary".format(type(self).__name__))
        if self.__dict__.get('method') is None:
            raise ValueError("{} requires 'method' key in config dictionary".format(type(self).__name__))
        self._selection = self._selection_method(self.method)

    @staticmethod
    def _selection_method(selection: str) -> Callable:
        methods = {
            'basic': UniformSelectionTransform._basic_selection,
            'average': UniformSelectionTransform._average_selection,
            'max': UniformSelectionTransform._max_selection
        }
        return methods[selection]

    @staticmethod
    def _pad_blob(blob: np.ndarray, scale: int) -> np.ndarray:
        x, y, z = blob.shape
        new_shape = [val + (scale - val % scale) % scale for val in (x, y, z)]
        padded_blob = np.zeros(new_shape, dtype=np.float32)
        padded_blob[:x, :y, :z] += blob
        return padded_blob

    @staticmethod
    def _nonzero(blob: np.ndarray) -> int:
        nonzero = np.array(np.nonzero(blob))
        return nonzero.shape[-1]

    @staticmethod
    def _basic_selection(blob: np.ndarray, scale: int) -> np.ndarray:
        voxel_samples = blob[scale // 2::scale, scale // 2::scale, scale // 2::scale]
        processed_blob = np.zeros(blob.shape)
        processed_blob[scale // 2::scale, scale // 2::scale, scale // 2::scale] = voxel_samples
        return processed_blob

    @staticmethod
    def _average_selection(blob: np.ndarray, scale: int) -> np.ndarray:
        padded_blob = UniformSelectionTransform._pad_blob(blob, scale)
        sub_arrays = []
        for (x, y, z) in itertools.product(list(range(scale)), repeat=3):
            sub_array = padded_blob[x::scale, y::scale, z::scale]
            sub_arrays.append(sub_array)
        sub_arrays = np.stack(sub_arrays)
        voxel_samples = np.average(sub_arrays, axis=0)
        processed_blob = np.zeros(padded_blob.shape)
        processed_blob[scale // 2::scale, scale // 2::scale, scale // 2::scale] = voxel_samples
        processed_blob = processed_blob[:blob.shape[0], :blob.shape[1], :blob.shape[2]]
        return processed_blob

    @staticmethod
    def _max_selection(blob: np.ndarray, scale: int) -> np.ndarray:
        padded_blob = UniformSelectionTransform._pad_blob(blob, scale)
        sub_arrays = []
        for (x, y, z) in itertools.product(list(range(scale)), repeat=3):
            sub_array = padded_blob[x::scale, y::scale, z::scale]
            sub_arrays.append(sub_array)
        sub_arrays = np.stack(sub_arrays)
        voxel_samples = np.max(sub_arrays, axis=0)
        processed_blob = np.zeros(padded_blob.shape)
        processed_blob[scale // 2::scale, scale // 2::scale, scale // 2::scale] = voxel_samples
        processed_blob = processed_blob[:blob.shape[0], :blob.shape[1], :blob.shape[2]]
        return processed_blob

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        nonzero = self._nonzero(blob)
        if nonzero <= self.max_voxel:
            return blob
        scale = (nonzero / self.max_voxel)**(1/3)
        scale = math.ceil(scale)
        processed_blob = blob
        while self._nonzero(processed_blob) > self.max_voxel:
            processed_blob = self._selection(blob, scale)
            scale += 1
        return processed_blob


TRANSFORMS = {
    "BlobSurfaceTransform": BlobSurfaceTransform
}

# the function is left only for the sake of development.
# TODO: Remove before merging to the `main`
if __name__ == "__main__":
    files = os.listdir("../../data/blobs_full")[12]
    blob = np.load(f"../../data/blobs_full/{files}")["blob"]
    print("Pre-size: {}".format(UniformSelectionTransform._nonzero(blob)))
    transform = UniformSelectionTransform({'max_voxel': 2000, 'method': 'max'})
    transformed = transform.preprocess(blob)
    print("Post-size: {}".format(UniformSelectionTransform._nonzero(transformed)))
