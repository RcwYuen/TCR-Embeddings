from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
)
from pathlib import Path
import pandas as pd
import torch
import numpy as np

dir = Path(__file__).resolve().parent
tokenizer = AutoTokenizer.from_pretrained(dir / "tcrbert-tokenizer")
bertmodel = AutoModelForMaskedLM.from_pretrained(dir / "tcrbert-model").bert
bertmodel = bertmodel.to("cuda") if torch.cuda.is_available() else bertmodel


def calc_vector_representations(df, batchsize = 2 ** 10):
    tcr = [i for i in df["CDR3A"].tolist() + df["CDR3B"].tolist() if not pd.isna(i)]
    loader = torch.utils.data.DataLoader(tcr, batch_size = batchsize, shuffle = False)
    embeddings = []
    for i in loader:
        inputs = tokenizer(i, return_tensors = "pt", padding = True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        e = bertmodel(**inputs).last_hidden_state
        embeddings += torch.mean(e, dim=1).tolist()
    return np.array(embeddings)