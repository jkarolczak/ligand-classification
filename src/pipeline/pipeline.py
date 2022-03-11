from typing import List
from pipeline.transforms import *


class Pipeline:
    """
    Class that allows chaining of Transforms
    """

    def __init__(self,
                 transforms: List[Transform] = None,
                 col_name: str = None) -> None:
        self._transforms = transforms
        self.col_name = col_name

    @staticmethod
    def elementwise_len(series: pd.Series) -> pd.Series:
        return series.map(lambda lst: len(lst))

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, value: List[Transform]):
        self._transforms = value

# perhaps some other functionality


if __name__ == "__main__":
    pass
