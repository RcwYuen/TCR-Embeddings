import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tcr_embeddings import runtime_constants

os.chdir(runtime_constants.HOME_PATH)
sys.path.append(str(runtime_constants.HOME_PATH))
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")


def generate(dim=5, method=lambda shape: np.random.uniform(0, 1, size=shape)):
    colnames = [f"f.{i}" for i in range(dim)]
    embeddings = pd.DataFrame(
        method((len(amino_acids), 5)), index=amino_acids, columns=colnames
    )
    embeddings.index.name = "amino.acid"
    embeddings.to_csv(Path(__file__).resolve().parent / "random.txt", sep="\t")


if __name__ == "__main__":
    generate()
