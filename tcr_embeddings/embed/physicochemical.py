import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tcr_embeddings import runtime_constants
from tcr_embeddings.embed.embedder import Embedder

os.chdir(runtime_constants.HOME_PATH)
sys.path.append(str(runtime_constants.HOME_PATH))


class PhysicoChemicalEncoder(Embedder):
    def __init__(self, fname: str) -> None:
        loc = Path(__file__).resolve().parent
        self.embedding_space = self._get_df(loc / fname)
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

    def _get_df(self, fname: str | Path) -> pd.DataFrame:
        df = pd.read_csv(fname, sep="\t").set_index("amino.acid").sort_index()
        df = (df.max() - df) / (df.max() - df.min())
        return df

    def get_out_dim(self) -> int:
        return self.embedding_space.shape[1]

    def calc_vector_representations(
        self, df: pd.DataFrame, *args, **kwargs
    ) -> np.ndarray:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The embedding tensor is already moved to GPU in __init__
        embedding_tensor = self.embedding_tensor

        # Combine and clean the sequences
        df_tcrs = pd.concat([df["CDR3A"].dropna(), df["CDR3B"].dropna()])
        df_tcrs = df_tcrs.apply(
            lambda tcr: "".join(filter(self.amino_acid_to_index.__contains__, tcr))
        )

        # Initialize the representation matrix
        rep = torch.zeros(
            (len(df_tcrs), embedding_tensor.shape[1]), dtype=torch.float32, device=device
        )

        # Preprocess sequences on the CPU
        tcr_indices = [
            torch.tensor(
                [self.amino_acid_to_index[char] for char in tcr], dtype=torch.long
            )
            for tcr in df_tcrs
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
