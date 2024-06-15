from pathlib import Path
import numpy as np
import random

def create_folds(files, k=5):
    files = [i for i in files]
    random.shuffle(files)
    fold_size = int(np.ceil(len(files) / k))
    return [files[i:i + fold_size] for i in range(0, len(files), fold_size)]

dir = Path(__file__).resolve().parent

k = 5
to_create = [
    dir / "tcvhcw/cleaned",
    dir / "Tx/cleaned"
]

for path in to_create:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    allfiles = list(path.glob("*.tsv"))
    folds = create_folds(allfiles, k = k)
    with open(path / "kfold.txt", "w") as f:
        for fold in folds:
            f.writelines("<>".join([str(i.relative_to(Path.cwd())) for i in fold]) + "\n")
    
