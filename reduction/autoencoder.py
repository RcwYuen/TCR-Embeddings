from pathlib import Path
from dotenv import load_dotenv
import os, sys

dir = Path(__file__).resolve().parent
load_dotenv(Path.cwd() / ".env")
python_path = os.getenv('PYTHONPATH')
if python_path:
    sys.path.append(python_path)

from embed.physicochemical import kidera, atchley, rand, aaprop
from embed.llm import tcrbert
from sceptr import variant
from reduction import AutoEncoder
from dotenv import load_dotenv

if __name__ == "__main__":
    dataset_paths = sum(
        [list((Path.cwd() / "data/tcvhcw/cleaned").glob("*.tsv")),
        list((Path.cwd() / "data/Tx/cleaned").glob("*.tsv"))], []
    )

    methods = [(atchley(), "atchley"), (aaprop(), "aaprop"), (rand(), "rand"), (kidera(), "kidera"), (tcrbert(), "tcrbert"), (variant.default(), "sceptr-default"), (variant.tiny(), "sceptr-tiny")]

    for method in methods:
        ae = AutoEncoder(*method, out_dim = 5, batchsize = 4096)
        ae.create_transformation(dataset_paths)