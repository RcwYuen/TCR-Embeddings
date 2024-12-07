import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from tcr_embeddings import runtime_constants
from tcr_embeddings.embed._embedder import Embedder

os.chdir(runtime_constants.HOME_PATH)
sys.path.append(str(runtime_constants.HOME_PATH))


class TCRDataset(torch.utils.data.Dataset):
    def __init__(self, ls_tcr):
        super().__init__()
        self.data = ls_tcr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class LLMEncoder(Embedder):
    def __init__(self, tokenizer, model):
        self._tokenizer = tokenizer
        self._model = model

    def calc_vector_representations(self, df, batchsize=2**10):
        tcr_dataset = TCRDataset(
            [
                " ".join(list(i))
                for i in df["CDR3A"].tolist() + df["CDR3B"].tolist()
                if not pd.isna(i)
            ]
        )
        loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            tcr_dataset, batch_size=batchsize, shuffle=False
        )
        embeddings = []
        for i in loader:
            inputs = self._tokenizer(i, return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            embeddings += torch.mean(
                self._model(**inputs).last_hidden_state, dim=1
            ).tolist()
        return np.array(embeddings)

    def _get_df(self, fname: str):
        pass


def tcrbert() -> LLMEncoder:
    tokenizer = AutoTokenizer.from_pretrained(
        Path(__file__).resolve().parent / "tcrbert-tokenizer"
    )
    bertmodel = AutoModelForMaskedLM.from_pretrained(
        Path(__file__).resolve().parent / "tcrbert-model"
    ).bert
    bertmodel = bertmodel.to("cuda") if torch.cuda.is_available() else bertmodel
    return LLMEncoder(tokenizer, bertmodel)
