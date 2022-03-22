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

    :param config: has to contain key describing the neighbourhood, one of 6, 22 or 26
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
        blob = generic_filter(
            input=blob,
            function=BlobSurfaceTransform._filter,
            footprint=self._footprint(),
            mode="constant",
            cval=0.0
        )
        return blob


TRANSFORMS = {
    "BlobSurfaceTransform": BlobSurfaceTransform
}
