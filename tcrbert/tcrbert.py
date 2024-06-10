from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
)
from pathlib import Path
import pandas as pd
import torch

tokenizer = AutoTokenizer.from_pretrained(Path.cwd() / "tokenizer")
bertmodel = AutoModelForMaskedLM.from_pretrained(Path.cwd() / "model").bert
bertmodel = bertmodel.to("cuda") if torch.cuda.is_available() else bertmodel


def embed(df, batchsize = 2 ** 10):
    tcr = [i for i in df["CDR3A"].tolist() + df["CDR3B"].tolist() if not pd.isna(i)]
    loader = torch.utils.data.DataLoader(tcr, batch_size = batchsize, shuffle = False)

