from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans

from read_blob import BlobDetailsA

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
_CLUSTERING = "Clustering"


class Transform(ABC):
    """
    abstract class for preprocessing transformations
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


class Clustering(Transform):
    """
    class for limiting the number of points in blob using k-means clustering
    """

    def __init__(self, blob: np.ndarray = None, num_points: int = 2000, **kwargs) -> None:
        super().__init__(blob, **kwargs)
        self.blob = blob
        self.num_points = num_points
        self.name = _CLUSTERING

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        """
        performs kmeans clustering on given blob
        :param blob: blob to be processed
        :return: numpy array, processed blob with nonzero points being clusters' centers
        """
        X = self.extract_info(blob)
        n_clusters = min(X.shape[0], self.num_points)
        kmeans = KMeans(n_clusters=n_clusters, random_state=23).fit(X)
        new_blob = self.create_new_blob(blob.shape, kmeans.cluster_centers_)
        return new_blob

    def extract_info(self, blob: np.ndarray) -> np.ndarray:
        """
        extracts information from raw blob representation
        :param blob: blob to be processed
        :return: numpy array of data prepared for clustering - coordinates and features
        """
        coordinates = np.transpose(np.nonzero(blob))
        features = blob[np.nonzero(blob)]
        features = np.expand_dims(features, axis=1)
        return np.concatenate((coordinates, features), axis=1)

    def create_new_blob(self, shape: Tuple[int], cluster_centers: np.ndarray) -> np.ndarray:
        """
        creates new blob based on the obtained cluster centers
        :param shape: tuple of integers, shape of the original blob
        :param cluster_centers: numpy array, obtained cluster centers
        :return: numpy array, the created blob
        """
        new_blob = np.zeros(shape)
        split = np.hsplit(cluster_centers, np.array([3, 6]))
        coordinates, features = split[0], split[1]
        coordinates = np.around(coordinates).astype(int)
        coordinates = tuple(np.transpose(coordinates))
        features = np.squeeze(features, axis=1)
        new_blob[coordinates] = features
        return new_blob
