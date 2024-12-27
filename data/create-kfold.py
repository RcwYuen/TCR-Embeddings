import random
from pathlib import Path

import numpy as np


def create_folds(files, k=5):
    files = [i for i in files]
    random.shuffle(files)
    fold_size = int(np.ceil(len(files) / k))
    return [files[i : i + fold_size] for i in range(0, len(files), fold_size)]


location = Path(__file__).resolve().parent

k = 5
to_create = [location / "tcvhcw/cleaned", location / "Tx/cleaned"]

for path in to_create:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    allfiles = list(path.glob("*.tsv"))
    folds = create_folds(allfiles, k=k)
    with open(path / "kfold.txt", "w") as f:
        for fold in folds:
            f.writelines("<>".join([str(i.name) for i in fold]) + "\n")
