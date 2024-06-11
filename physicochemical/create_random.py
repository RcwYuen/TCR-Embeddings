import pandas as pd
import numpy as np

amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

def generate(dim = 5, method = lambda shape: np.random.uniform(0, 1, size = shape)):
    colnames = [f"f.{i}" for i in range(dim)]
    embeddings = pd.DataFrame(method((len(amino_acids), 5)), index = amino_acids, columns = colnames)
    embeddings.to_csv("random.txt", sep = "\t")
