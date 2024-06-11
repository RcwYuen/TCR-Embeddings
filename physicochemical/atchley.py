import pandas as pd
import numpy as np
from pathlib import Path

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

global embedding_space
dir = Path(__file__).resolve().parent
embedding_space = _get_df(dir / "atchley.txt")