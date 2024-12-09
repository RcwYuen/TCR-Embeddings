import json
from pathlib import Path

from tcr_embeddings.training.training_utils import load_default_configs

loc = Path(__file__).resolve().parent

encodings = [
    "atchley",
    "kidera",
    "rand",
    "aaprop",
    "tcrbert",
    "sceptr-default",
    "sceptr-tiny",
]
kfolds = 5

if __name__ == "__main__":
    config = load_default_configs()

    for encoding in encodings:
        for fold in range(kfolds):
            config["encoding"] = encoding
            config["kfold"] = fold
            config["output-path"] = f"results/{encoding}-no-reduction/kfold-{fold}"
            config["reduction"] = ""
            with open(loc / f"{encoding}-kfold-{fold}-no-reduction.json", "w") as f:
                f.write(json.dumps(config, indent=4))

            config["reduction"] = "autoencoder"
            config["output-path"] = f"results/{encoding}-autoencoder/kfold-{fold}"

            with open(loc / f"{encoding}-kfold-{fold}-autoencoder.json", "w") as f:
                f.write(json.dumps(config, indent=4))
