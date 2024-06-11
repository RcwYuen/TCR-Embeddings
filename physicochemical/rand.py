import pandas as pd
import numpy as np
from pathlib import Path
from physicochemical import create_random

def _get_df(fname):
    df = pd.read_csv(fname, delimiter = "\t").set_index("amino.acid").sort_index()
    df = (df.max() - df) / (df.max() - df.min())
    return df


def calc_vector_representations(repertoire, batchsize = None):
    df = embedding_space.copy()
    tcrs = [i for i in repertoire["CDR3A"].tolist() + repertoire["CDR3B"].tolist() if not pd.isna(i)]
    rep = []
    for tcr in tcrs:
        if tcr is not None:
            tcr = ''.join(filter(lambda char: char in df.index.tolist(), tcr))
            if tcr != "":
                rep.append(np.mean(df.loc[list(tcr)].values, axis = 0))
    rep = np.array(rep)
    rep = rep[~np.isnan(rep).any(axis = 1)]
    return rep

def recreate(dim = 5, method = lambda shape: np.random.uniform(0, 1, size = shape)):
    create_random.generate(dim = dim, method = method)

global embedding_space
dir = Path(__file__).resolve().parent
embedding_space = _get_df(dir / "random.txt")