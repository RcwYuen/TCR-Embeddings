from pathlib import Path

import torch.nn
import pandas as pd

HOME_PATH: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = Path(__file__).resolve().parent.parent / "data"
CRITERION: torch.nn.Module = torch.nn.BCELoss()
DF_SAMPLE: pd.DataFrame = pd.read_csv(DATA_PATH / "sample.tsv", sep = "\t")
DF_FULL: pd.DataFrame = pd.read_csv(DATA_PATH / "full.tsv", sep = "\t")