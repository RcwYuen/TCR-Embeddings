import os
import sys
from pathlib import Path

from dotenv import load_dotenv

dir = Path(__file__).resolve().parent
load_dotenv(Path.cwd() / ".env")
python_path = os.getenv("PYTHONPATH")
if python_path:
    sys.path.append(python_path)

import pandas as pd
import torch
import torch.nn.functional as F

from tcr_embeddings.embed._embedder import Embedder


class PhysicoChemicalEncoder(Embedder):
    def __init__(self, fname):
        dir = Path(__file__).resolve().parent
        self.embedding_space = self._get_df(dir / fname)
        self.amino_acid_to_index = {
            aa: i for i, aa in enumerate(self.embedding_space.index)
        }
        self.embedding_tensor = torch.tensor(
            self.embedding_space.values, dtype=torch.float32
        )
        self.embedding_tensor = (
            self.embedding_tensor.cuda()
            if torch.cuda.is_available()
            else self.embedding_tensor
        )

    def _get_df(self, fname):
        df = pd.read_csv(fname, sep="\t").set_index("amino.acid").sort_index()
        df = (df.max() - df) / (df.max() - df.min())
        return df

    def calc_vector_representations(self, df, *args, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The embedding tensor is already moved to GPU in __init__
        embedding_tensor = self.embedding_tensor

        # Combine and clean the sequences
        tcrs = pd.concat([df["CDR3A"].dropna(), df["CDR3B"].dropna()])
        tcrs = tcrs.apply(
            lambda tcr: "".join(filter(self.amino_acid_to_index.__contains__, tcr))
        )

        # Initialize the representation matrix
        rep = torch.zeros(
            (len(tcrs), embedding_tensor.shape[1]), dtype=torch.float32, device=device
        )

        # Preprocess sequences on the CPU
        tcr_indices = [
            torch.tensor(
                [self.amino_acid_to_index[char] for char in tcr], dtype=torch.long
            )
            for tcr in tcrs
            if tcr
        ]

        # Transfer all indices to GPU in one go
        tcr_indices = [indices.to(device) for indices in tcr_indices]

        # Process each TCR sequence on the GPU
        for i, indices in enumerate(tcr_indices):
            if indices.numel() > 0:
                embeddings = F.embedding(indices, embedding_tensor)
                rep[i] = embeddings.mean(dim=0)
            else:
                rep[i] = torch.zeros(
                    embedding_tensor.shape[1], dtype=torch.float32, device=device
                )

        return rep.cpu().numpy()  # Convert back to numpy array if needed


def aaprop() -> PhysicoChemicalEncoder:
    return PhysicoChemicalEncoder("aa_properties.txt")


def atchley() -> PhysicoChemicalEncoder:
    return PhysicoChemicalEncoder("atchley.txt")


def rand() -> PhysicoChemicalEncoder:
    return PhysicoChemicalEncoder("random.txt")


def kidera() -> PhysicoChemicalEncoder:
    return PhysicoChemicalEncoder("kidera.txt")
