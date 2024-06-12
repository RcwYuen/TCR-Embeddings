from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
)
from pathlib import Path

dir = Path(__file__).resolve().parent
tokenizer = AutoTokenizer.from_pretrained("wukevin/tcr-bert-mlm-only")
model = AutoModelForMaskedLM.from_pretrained("wukevin/tcr-bert-mlm-only")
model.save_pretrained(dir / "tcrbert/model/")
tokenizer.save_pretrained(dir / "tcrbert/tokenizer/")
tokenizer.save_vocabulary(dir / "tcrbert/tokenizer/")