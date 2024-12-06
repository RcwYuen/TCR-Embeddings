import os
import sys
from pathlib import Path

from dotenv import load_dotenv

dir = Path(__file__).resolve().parent
load_dotenv(Path.cwd() / ".env")
python_path = os.getenv("PYTHONPATH")
if python_path:
    sys.path.append(python_path)

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from embed._embedder import Embedder


class LLMEncoder(Embedder):
    def __init__(self, tokenizer, model):
        self._tokenizer = tokenizer
        self._model = model

    def calc_vector_representations(self, df, batchsize=2**10):
        tcr = [
            " ".join(list(i))
            for i in df["CDR3A"].tolist() + df["CDR3B"].tolist()
            if not pd.isna(i)
        ]
        loader = torch.utils.data.DataLoader(tcr, batch_size=batchsize, shuffle=False)
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
    tokenizer = AutoTokenizer.from_pretrained(dir / "tcrbert-tokenizer")
    bertmodel = AutoModelForMaskedLM.from_pretrained(dir / "tcrbert-model").bert
    bertmodel = bertmodel.to("cuda") if torch.cuda.is_available() else bertmodel
    bertmodel = torch.nn.ModuleList(
        [bertmodel.embeddings] + [layer for layer in bertmodel.encoder.layer[:8]]
    )
    return LLMEncoder(tokenizer, bertmodel)
