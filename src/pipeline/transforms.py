import os
from abc import ABC, abstractmethod
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy.ndimage import zoom
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
    A class that limits the number of voxels in the blob by averaging densities within 2x2x2 segments if the number
    of voxels within the original blob exceeds the max_voxel. This class extends 'Transformer' class.

    Requires 'max_voxel' parameter to be defined within the provided config dictionary.
    """

    def __init__(self, config: Union[Dict, None] = None, **kwargs) -> None:
        super().__init__(config, **kwargs)
        if not self.__dict__.get('max_voxel'):
            raise ValueError("{} requires 'max_voxel' key in config dictionary".format(type(self).__name__))

    @staticmethod
    def _pad_blob(blob: np.ndarray) -> np.ndarray:
        x, y, z = blob.shape
        new_shape = (x + x % 2, y + y % 2, z + z % 2)
        padded_blob = np.zeros(new_shape, dtype=np.float32)
        padded_blob[:x, :y, :z] += blob
        return padded_blob

    @staticmethod
    def _nonzero(blob: np.ndarray) -> int:
        nonzero = np.array(np.nonzero(blob))
        return nonzero.shape[-1]

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        if self._nonzero(blob) <= self.max_voxel:
            return blob
        padded_blob = self._pad_blob(blob)
        sub_arrays = []
        for (x, y, z) in itertools.product((0, 1), repeat=3):
            sub_array = padded_blob[x::2, y::2, z::2]
            sub_arrays.append(sub_array)
        sub_arrays = np.stack(sub_arrays)
        processed_blob = np.average(sub_arrays, axis=0)
        return processed_blob


class InterpolationTransform(Transform):
    """
    A class that limits the number of voxels to the desired number of nonzero voxels by using spline interpolation,
    This class extends 'Transformer' class.

    Requires 'max_voxel' parameter to be defined within the provided config dictionary.

    TODO: Verify correctness?
    """

    def __init__(self, config: Union[Dict, None] = None, **kwargs) -> None:
        super().__init__(config, **kwargs)
        if not self.__dict__.get('max_voxel'):
            raise ValueError("{} requires 'max_voxel' key in config dictionary".format(type(self).__name__))

    @staticmethod
    def _nonzero(blob: np.ndarray) -> int:
        nonzero = np.array(np.nonzero(blob))
        return nonzero.shape[-1]

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        nonzero = self._nonzero(blob)
        if nonzero <= self.max_voxel:
            return blob
        scale = self.max_voxel / nonzero
        processed_blob = zoom(blob, scale, order=1, mode='nearest', grid_mode=True)
        return processed_blob


TRANSFORMS = {
    "BlobSurfaceTransform": BlobSurfaceTransform
}

# the function is left only for the sake of development.
# TODO: Remove before merging to the `main`
if __name__ == "__main__":
    files = os.listdir("../../data/blobs_full")[0]
    blob = np.load(f"../../data/blobs_full/{files}")["blob"]
    transform = ...
    transformed = transform.preprocess(blob)
