from sceptr import sceptr
import pandas as pd
from src.model import sceptr_unidirectional, load_trained
from pathlib import Path
import torch
import re
import warnings
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")

dir = Path.cwd() / "results" / "sceptr" / "trained-sceptr-caneval-4"
#pattern = re.compile(r"eval-set-auc-(.*).csv")
#bestepoch = int(pattern.match(str(list(dir.glob("eval-set-auc-*.csv"))[0].name)).group(1))
bestepoch = 49

model = dir / f"Epoch {bestepoch}" / f"classifier-{bestepoch}.pth"
model = load_trained(model, sceptr_unidirectional)