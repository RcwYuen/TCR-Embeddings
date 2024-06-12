import pandas as pd
import numpy as np
from pathlib import Path

dir = Path(__file__).resolve().parent
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

def generate(dim = 5, method = lambda shape: np.random.uniform(0, 1, size = shape)):
    colnames = [f"f.{i}" for i in range(dim)]
    embeddings = pd.DataFrame(method((len(amino_acids), 5)), index = amino_acids, columns = colnames)
    embeddings.index.name = "amino.acid"
    embeddings.to_csv(dir / "random.txt", sep = "\t")

if __name__ == "__main__":
    generate()