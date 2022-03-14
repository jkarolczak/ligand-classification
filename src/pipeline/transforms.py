from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

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
