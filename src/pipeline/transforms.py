from abc import ABC, abstractmethod

from typing import Callable, Dict, Tuple

from typing import Tuple

import numpy as np
import itertools
import math
from skimage.measure import marching_cubes

from scipy.ndimage import generic_filter
from sklearn.cluster import KMeans

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
    def _filter(neighbourhood: np.ndarray) -> float:
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


class ClusteringTransform(Transform):
    """
    A class that limits the number of voxels in the blob using k-means clustering.

    :param config: has to contain "max_blob_size" (maximal number of nonzero voxels in the blob)
    """

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        """
        performs kmeans clustering on given blob
        :param blob: blob to be processed
        :type blob: np.ndarray
        :returns: processed blob with number of voxels limited to max_blob_size. If the value of "max_blob_size" is
        greater than the number of nonzero voxels in blob, new blob with nonzero voxels being clusters' centers is
        returned. Otherwise, when the value of "max_blob_size" is less or equal to the number of nonzero voxels in the
        blob, there are no modifications made and the whole blob is returned.
        :return type: np.ndarray
        """
        if self.__dict__.get('max_blob_size') is None:
            raise ValueError("{} requires 'max_blob_size' key in config dictionary".format(type(self).__name__))
        coordinates = np.transpose(np.nonzero(blob))
        features = blob[np.nonzero(blob)]
        if coordinates.shape[0] <= self.max_blob_size:
            return blob
        else:
            kmeans = KMeans(n_clusters=self.max_blob_size, random_state=23).fit(coordinates)
            features_clusters = np.array([features[kmeans.labels_ == i].mean() for i in np.unique(kmeans.labels_)])
            new_blob = self._create_new_blob(blob.shape, kmeans.cluster_centers_, features_clusters)
            return new_blob

    @staticmethod
    def _create_new_blob(shape: Tuple[int], coordinates: np.ndarray, features: np.ndarray) -> np.ndarray:
        new_blob = np.zeros(shape)
        coordinates = np.around(coordinates).astype(int)
        coordinates = tuple(np.transpose(coordinates))
        new_blob[coordinates] = features
        return new_blob


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
        if self.__dict__.get('max_blob_size') is None:
            raise ValueError("{} requires 'max_blob_size' key in config dictionary".format(type(self).__name__))
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


class UniformSelectionTransform(Transform):
    """
    A class that limits the number of voxels in the blob sampling only the middle voxels within n x n x n blocks.
    The value assigned to selected voxels depends on the value of 'method' in config dictionary.
    This class extends 'Transformer' class.

    :param config: configuration dictionary with integer 'max_blob_size' (maximal number of remaining voxels)
     and string 'method' (dictating the method of assigning values to sampled voxels - options: 'basic'/'average'/'max')
    """

    def __init__(self, config: Union[Dict, None] = None, **kwargs) -> None:
        super().__init__(config, **kwargs)
        if self.__dict__.get('max_blob_size') is None:
            raise ValueError("{} requires 'max_blob_size' key in config dictionary".format(type(self).__name__))
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
        if nonzero <= self.max_blob_size:
            return blob
        scale = (nonzero / self.max_blob_size) ** (1 / 3)
        scale = math.ceil(scale)
        processed_blob = blob
        while self._nonzero(processed_blob) > self.max_blob_size:
            processed_blob = self._selection(blob, scale)
            scale += 1
        return processed_blob


TRANSFORMS = {
    "BlobSurfaceTransform": BlobSurfaceTransform,
    "ClusteringTransform": ClusteringTransform,
    "RandomSelectionTransform": RandomSelectionTransform,
    "UniformSelectionTransform": UniformSelectionTransform
}
