import os
import sys
import warnings
from pathlib import Path

from reduction import AutoEncoder
from sceptr import variant

from tcr_embeddings import runtime_constants
from tcr_embeddings.embed.llm import tcrbert
from tcr_embeddings.embed.physicochemical import aaprop, atchley, kidera, rand

os.chdir(runtime_constants.HOME_PATH)
sys.path.append(str(runtime_constants.HOME_PATH))


if __name__ == "__main__":
    dataset_paths = list((Path.cwd() / "data/tcvhcw/cleaned").glob("*.tsv")) + list(
        (Path.cwd() / "data/Tx/cleaned").glob("*.tsv")
    )

    methods = [
        (atchley(), "atchley"),
        (aaprop(), "aaprop"),
        (rand(), "rand"),
        (kidera(), "kidera"),
        (tcrbert(), "tcrbert"),
        (variant.default(), "sceptr-default"),
        (variant.tiny(), "sceptr-tiny"),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in methods:
            ae = AutoEncoder(*method, out_dim=5, batchsize=4096)
            ae.create_transformation(dataset_paths)
