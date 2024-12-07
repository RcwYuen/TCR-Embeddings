import os
import sys
from pathlib import Path

from dotenv import load_dotenv

dir = Path(__file__).resolve().parent
load_dotenv(Path.cwd() / ".env")
python_path = os.getenv("PYTHONPATH")
if python_path:
    sys.path.append(python_path)

import os

from transformers import AutoModelForMaskedLM, AutoTokenizer

dir = Path(__file__).resolve().parent

files = list((dir / "tcrbert-model").glob("*")) + list(
    (dir / "tcrbert-tokenizer").glob("*")
)

for file in files:
    os.remove(file)

tokenizer = AutoTokenizer.from_pretrained("wukevin/tcr-bert-mlm-only")
model = AutoModelForMaskedLM.from_pretrained("wukevin/tcr-bert-mlm-only")
model.save_pretrained(dir / "tcrbert-model")
tokenizer.save_pretrained(dir / "tcrbert-tokenizer")
tokenizer.save_vocabulary(dir / "tcrbert-tokenizer")
