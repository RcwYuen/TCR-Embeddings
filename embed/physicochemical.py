import pandas as pd
import numpy as np
from pathlib import Path
from embed._embedder import Embedder
from dotenv import load_dotenv
import os, sys

load_dotenv()
python_path = os.getenv('PYTHONPATH')
if python_path:
    sys.path.append(python_path)


class PhysicoChemicalEncoder(Embedder):
    def __init__(self, fname):
        dir = Path(__file__).resolve().parent
        self.embedding_space = self._get_df(dir / fname)

    def _get_df(self, fname):
        df = pd.read_csv(fname, delimiter = "\t").set_index("amino.acid").sort_index()
        df = (df.max() - df) / (df.max() - df.min())
        return df
    
    def calc_vector_representations(self, df, *args, **kwargs):
        embedding_space = self.embedding_space.copy()
        tcrs = [i for i in df["CDR3A"].tolist() + df["CDR3B"].tolist() if not pd.isna(i)]
        rep = []
        for tcr in tcrs:
            if tcr is not None:
                tcr = ''.join(filter(lambda char: char in embedding_space.index.tolist(), tcr))
                if tcr != "":
                    rep.append(np.mean(embedding_space.loc[list(tcr)].values, axis = 0))
        rep = np.array(rep)
        rep = rep[~np.isnan(rep).any(axis = 1)]
        return rep
    
def aaprop():
    return PhysicoChemicalEncoder("aa_properties.txt")

def atchley():
    return PhysicoChemicalEncoder("atchley.txt")

def rand():
    return PhysicoChemicalEncoder("random.txt")

def kidera():
    return PhysicoChemicalEncoder("kidera.txt")