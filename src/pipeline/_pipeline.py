from typing import List, Dict

import numpy as np

from pipeline.transforms import *


class Pipeline:
    """
    Class that allows chaining of Transforms
    """

    def __init__(self, steps: List[Dict]) -> None:
        self._transforms = [TRANSFORMS[s["name"]](s["config"]) for s in steps]

    @property
    def transforms(self) -> List[Transform]:
        """
        getter for Pipeline
        :return: List of Transformations applied to a blob
        """
        return self._transforms

    @transforms.setter
    def transforms(self, value: List[Transform]):
        self._transforms = value

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        for _t in self._transforms:
            blob = _t.preprocess(blob)
        return blob


if __name__ == "__main__":
    pass
