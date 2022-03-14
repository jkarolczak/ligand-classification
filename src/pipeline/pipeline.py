from typing import List
from pipeline.transforms import *


class Pipeline:
    """
    Class that allows chaining of Transforms
    """

    def __init__(self,
                 transforms: List[Transform] = None,
                 blob: np.ndarray = None) -> None:
        self._transforms = transforms
        self.blob = blob

    @transforms.getter
    def transforms(self) -> List[Transform]:
        """
        getter for Pipeline
        :return: List of Transformations applied to a blob
        """
        return self._transforms

    @transforms.setter
    def transforms(self, value: List[Transform]):
        self._transforms = value


# perhaps some other functionality


if __name__ == "__main__":
    pass
