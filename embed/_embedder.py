from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Embedder(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def calc_vector_representations(
        self, df: pd.DataFrame, batchsize: int = 2**10
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _get_df(self, fname: str) -> pd.DataFrame:
        pass
