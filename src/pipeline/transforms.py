import os
from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np
from scipy.ndimage import generic_filter

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

    :param config: dictionary containing transformation configuration. It's keys will be transformed to object fields.
    Example: config['foo'] = 'bar' -> transformation.foo = 'bar'
    """

    def __init__(self, config: Union[Dict, None] = None, **kwargs) -> None:
        if config:
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

    :param config: has to contain key "neighbourhood" defining the neighbourhood, one of 6, 22 or 26
    """

    def _footprint(self):
        assert int(self.neighbourhood) in [6, 22, 26]
        _masks = {
            6: np.array([
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
            ]),
            22: np.array([
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            ]),
            26: np.ones((3, 3, 3))
        }
        return _masks[int(self.neighbourhood)]

    @staticmethod
    def _filter(neighbourhood: np.ndarray) -> None:
        if np.any(neighbourhood == 0):
            return neighbourhood[int(neighbourhood.shape[0] / 2)]
        return 0.0

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        """
        :param blob: input blob
        :type blob: np.ndarray
        :returns: blob representing surface of the input blob
        :return type: np.ndarray
        """
        blob = generic_filter(
            input=blob,
            function=BlobSurfaceTransform._filter,
            footprint=self._footprint(),
            mode="constant",
            cval=0.0
        )
        return blob


class RandomSelectionTransform(Transform):
    """
    A class that limit voxels in the blob by drawing non-zero voxels. This class extends `Transformer` class.

    :param config: has to contain key "max_blob_size" specifying maximal number of voxels in the blob after drawing.
    """

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        """
        :param blob: input blob
        :type blob: np.ndarray
        :returns: blob that number of voxels was limited to "max_blob_size". When the blob has less non-zero voxels than
        "max_blob_size" whole blob, without modification is returned.
        :return type: np.ndarray
        """
        non_zeros = blob.nonzero()
        if non_zeros[0].shape[0] <= self.max_blob_size:
            return blob

        indices_mask = np.array(range(non_zeros[0].shape[0]))
        indices_mask = np.random.choice(indices_mask, size=self.max_blob_size, replace=False)
        x = non_zeros[0][indices_mask]
        y = non_zeros[1][indices_mask]
        z = non_zeros[2][indices_mask]

        mask = np.zeros_like(blob)
        mask[x, y, z] = 1.0

        return blob * mask


TRANSFORMS = {
    "BlobSurfaceTransform": BlobSurfaceTransform,
    "RandomSelectionTransform": RandomSelectionTransform
}
