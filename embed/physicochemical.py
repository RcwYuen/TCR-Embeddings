from pathlib import Path
from dotenv import load_dotenv
import os, sys

dir = Path(__file__).resolve().parent
load_dotenv(Path.cwd() / ".env")
python_path = os.getenv('PYTHONPATH')
if python_path:
    sys.path.append(python_path)

import pandas as pd
from embed._embedder import Embedder
import torch
import torch.nn.functional as F

class PhysicoChemicalEncoder(Embedder):
    def __init__(self, fname):
        dir = Path(__file__).resolve().parent
        self.embedding_space = self._get_df(dir / fname)
        self.amino_acid_to_index = {aa: i for i, aa in enumerate(self.embedding_space.index)}
        self.embedding_tensor = torch.tensor(self.embedding_space.values, dtype=torch.float32)

    def _get_df(self, fname):
        df = pd.read_csv(fname, delimiter="\t").set_index("amino.acid").sort_index()
        df = (df.max() - df) / (df.max() - df.min())
        return df

    def calc_vector_representations(self, df, *args, **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move embedding tensor to GPU
        embedding_tensor = self.embedding_tensor.to(device)

        # Combine and clean the sequences
        tcrs = pd.concat([df["CDR3A"].dropna(), df["CDR3B"].dropna()])
        tcrs = tcrs.apply(lambda tcr: ''.join(filter(self.amino_acid_to_index.__contains__, tcr)))

        # Initialize the representation matrix
        rep = torch.zeros((len(tcrs), embedding_tensor.shape[1]), dtype=torch.float32).to(device)

        # Process each TCR sequence
        for i, tcr in enumerate(tcrs):
            if tcr:
                indices = torch.tensor([self.amino_acid_to_index[char] for char in tcr], dtype=torch.long).to(device)
                embeddings = F.embedding(indices, embedding_tensor)
                rep[i] = embeddings.mean(dim=0)
            else:
                rep[i] = torch.zeros(embedding_tensor.shape[1], dtype=torch.float32).to(device)

        return rep.cpu().numpy()  # Convert back to numpy array if needed

def aaprop() -> PhysicoChemicalEncoder:
    return PhysicoChemicalEncoder("aa_properties.txt")

def atchley() -> PhysicoChemicalEncoder:
    return PhysicoChemicalEncoder("atchley.txt")

def rand() -> PhysicoChemicalEncoder:
    return PhysicoChemicalEncoder("random.txt")

def kidera() -> PhysicoChemicalEncoder:
    return PhysicoChemicalEncoder("kidera.txt")