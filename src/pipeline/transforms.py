import os
from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np
from skimage.measure import marching_cubes

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
        blob = marching_cubes(blob, 0)  # spacing=self.spacing, method=self.method)
        coords = blob[0].astype(int)
        shape = coords.max(-2)
        values = blob[3]
        blob = np.zeros(shape)
        np.put(blob, coords, values)
        return blob


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
