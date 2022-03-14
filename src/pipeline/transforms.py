from abc import ABC, abstractmethod

import numpy as np

"""
while writing our functions, we can specify the expected type of arguments and return. This is especially useful
for people browsing and reviewing our code. Almighty Python language offers some useful helpers:

using:

from typing import List, Tuple, Union

we can be more specific, e.g.
    def adam(grzenda: List[str]) -> Tuple[int]:

Union[str, List[str]] -> either string or a list of strings
"""

_EXAMPLE_CLASS = "Example class"


class Transform(ABC):
    """
    Abstract class for preprocessing transformations
    """

    def __init__(self, blob: np.ndarray = None, **kwargs) -> None:
        self.blob = blob

    @abstractmethod
    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        """
        Abstract method for applying preprocessing methods to a blob
        :param blob: blob to be processed
        """
        pass


class ExampleClass(Transform):
    """
    Example class showing all the best practices to follow during methods' implementation
    """

    def __init__(self, blob: np.ndarray = None, **kwargs) -> None:
        super().__init__(blob, **kwargs)
        self.blob = blob,
        self.name = _EXAMPLE_CLASS
        # other required attributes

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        """

        :param blob:
        """
        # should there be some method, that this transform uses, but could static, you can add @staticmethod annotation
        pass

    # define utility functions necessary


class BlobSurfaceTransform(Transform):
    """
    A class that limit voxels in the blob by removing all voxels that don't create surface of the blob. It removes
    voxels that don't have any 0 in their neighbourhood. This class extends `Transformer` class.
    """

    @staticmethod
    def _get_mask(blob: np.ndarray) -> np.ndarray:
        mask = np.ones_like(blob)
        blob = np.pad(blob, (1, 1), mode="constant", constant_values=0)
        for x in range(1, blob.shape[0] - 1):
            for y in range(1, blob.shape[1] - 1):
                for z in range(1, blob.shape[2] - 1):
                    neighbours = [blob[x - 1, y, z], blob[x + 1, y, z], blob[x, y - 1, z], blob[x, y + 1, z],
                                  blob[x, y, z - 1], blob[x, y, z + 1]]
                    neighbours = list(map(lambda _x: _x != 0, neighbours))
                    if all(neighbours):
                        mask[x - 1, y - 1, z - 1] = 0
        return mask

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        mask = self._get_mask(blob)
        return mask * blob
