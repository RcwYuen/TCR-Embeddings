import json
from pathlib import Path

import pandas as pd
import torch.nn

HOME_PATH: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = Path(__file__).resolve().parent.parent / "data"
CRITERION: torch.nn.Module = torch.nn.BCELoss()
DF_SAMPLE: pd.DataFrame = pd.read_csv(DATA_PATH / "sample.tsv", sep="\t")
DF_FULL: pd.DataFrame = pd.read_csv(DATA_PATH / "full.tsv", sep="\t")

PATH_POSITIVE_CLASS: list
PATH_NEGATIVE_CLASS: list
EPOCHS: int
LR: float | int
TRAIN_TEST_SPLIT: float | int
ACCUMMULATION: int
L2_PENALTY: float
RAND_SEED: int
USE_CUDA: bool

with open(HOME_PATH / "tcr_embeddings/constants.json") as f:
    _temp = json.load(f)
    globals().update(_temp)
    __all__ = list(_temp.keys())
