from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
)
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from dotenv import load_dotenv
import os, sys
from embed._embedder import Embedder

load_dotenv()
python_path = os.getenv('PYTHONPATH')
if python_path:
    sys.path.append(python_path)

dir = Path(__file__).resolve().parent

class LLMEmbedder(Embedder):
    def __init__(self, tokenizer, model):
        self._tokenizer = tokenizer
        self._model = model
    
    def calc_vector_representations(self, df, batchsize = 2 ** 10):
        tcr = [" ".join(list(i)) for i in df["CDR3A"].tolist() + df["CDR3B"].tolist() if not pd.isna(i)]
        loader = torch.utils.data.DataLoader(tcr, batch_size = batchsize, shuffle = False)
        embeddings = []
        for i in loader:
            inputs = self._tokenizer(i, return_tensors = "pt", padding = True)
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            embeddings += torch.mean(self._model(**inputs).last_hidden_state, dim=1).tolist()
        return np.array(embeddings)

    def _get_df(self, fname: str):
        pass

def tcrbert():
    tokenizer = AutoTokenizer.from_pretrained(dir / "tcrbert-tokenizer")
    bertmodel = AutoModelForMaskedLM.from_pretrained(dir / "tcrbert-model").bert
    bertmodel = bertmodel.to("cuda") if torch.cuda.is_available() else bertmodel
    return LLMEmbedder(tokenizer, bertmodel)