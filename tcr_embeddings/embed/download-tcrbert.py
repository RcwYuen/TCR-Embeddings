import os
import sys
from pathlib import Path

from transformers import AutoModelForMaskedLM, AutoTokenizer

from tcr_embeddings import runtime_constants

if __name__ == "__main__":
    os.chdir(runtime_constants.HOME_PATH)
    sys.path.append(str(runtime_constants.HOME_PATH))

    loc = Path(__file__).resolve().parent

    files = list((loc / "tcrbert-model").glob("*")) + list(
        (loc / "tcrbert-tokenizer").glob("*")
    )

    for file in files:
        os.remove(file)

    tokenizer = AutoTokenizer.from_pretrained("wukevin/tcr-bert-mlm-only")
    model = AutoModelForMaskedLM.from_pretrained("wukevin/tcr-bert-mlm-only")
    model.save_pretrained(str(loc / "tcrbert-model"))
    tokenizer.save_pretrained(str(loc / "tcrbert-tokenizer"))
    tokenizer.save_vocabulary(str(loc / "tcrbert-tokenizer"))
