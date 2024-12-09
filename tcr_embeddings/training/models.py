from pathlib import Path

import pandas as pd
import torch
from sparsemax import Sparsemax

from tcr_embeddings import constants
from tcr_embeddings.embed.embedder import Embedder


class MIClassifier(torch.nn.Module):
    def __init__(self, in_dim: int):
        super(MIClassifier, self).__init__()
        self.last_scores = None
        self.last_weights = None

        self.attention = torch.nn.Linear(in_dim, 1)
        self.classifying = torch.nn.Linear(in_dim, 1)
        self.sparsemax = Sparsemax(dim=0)
        self.relu = torch.nn.ReLU()
        self.sig = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_scores = self.attention(x)
        self.last_weights = self.sparsemax(self.last_scores.T).T  # type: ignore
        aggregated = torch.sum(self.last_weights * x, dim=0, keepdim=True)
        return self.sig(self.classifying(aggregated))


def reduced_classifier() -> MIClassifier:
    model = MIClassifier(5)
    return model.cuda() if torch.cuda.is_available() and constants.USE_CUDA else model


def ordinary_classifier(encoding_method: Embedder) -> MIClassifier:
    df = pd.read_csv(Path.cwd() / "data/sample.tsv", sep="\t", dtype=str)
    model = MIClassifier(encoding_method.calc_vector_representations(df).shape[1])
    return model.cuda() if torch.cuda.is_available() and constants.USE_CUDA else model
