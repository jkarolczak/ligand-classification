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

# TODO: we can use or reject using such definition of names,
# this way we may quite easy generate text of preprocessing steps -> Transform.name
# this can also be hardcoded in class or removed
_EXAMPLE_CLASS = "Example class"


class Transform(ABC):
    """
    abstract class for preprocessing transformations
    """

    # TODO: here we should probably debate, what to pass as the argument, so that it makes the most sense:
    #   - single blob ?
    #   - pd.DataFrame with blobs to be processed ?
    #       - this is my initial idea, we could apply transforms to entire pd.Series
    #   - something else?
    def __init__(self, col_name: str = None, **kwargs) -> None:
        self.col_name = col_name

    # TODO: discuss what to preprocess
    @abstractmethod
    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        pass

    # following my initial idea, I propose a method applying some preprocess to pd.Series
    def calculate(self, series: pd.Series) -> pd.Series:
        """
        Abstract method for applying preprocessing methods to Series
        Args:
            series (pd.Series): Series containing data (keywords, descriptions, etc.) to process
        Returns:
            pd.Series: Series containing processed data
        """
        if self.col_name is None:
            self.col_name = series.name
        series_tmp = series.map(self.preprocess)
        return series_tmp


class ExampleClass(Transform):
    """
    Example class showing all the best practices to follow during methods' implementation
    """

    def __init__(self, col_name: str = None, **kwargs) -> None:
        super().__init__(col_name, **kwargs)
        self.col_name = col_name,
        self.name = _EXAMPLE_CLASS
        # other required attributes

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        # implementation goes here or some utility methods
        # should there be some method, that this transform uses, but could static, you can add @staticmethod annotation
        pass

    # define utility functions necessary
