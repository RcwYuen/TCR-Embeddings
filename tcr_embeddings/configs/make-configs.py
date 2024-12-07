import json
from pathlib import Path

dir = Path(__file__).resolve().parent

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
config_path_base = Path.cwd() / "configs/_config.json"

with open(dir / config_path_base, "r") as file:
    config = json.load(file)

for encoding in encodings:
    for fold in range(kfolds):
        config["encoding"] = encoding
        config["kfold"] = fold
        config["output-path"] = f"results/{encoding}-no-reduction/kfold-{fold}"
        config["reduction"] = ""
        with open(dir / f"{encoding}-kfold-{fold}-no-reduction.json", "w") as f:
            f.write(json.dumps(config, indent=4))

        config["reduction"] = "autoencoder"
        config["output-path"] = f"results/{encoding}-autoencoder/kfold-{fold}"

        with open(dir / f"{encoding}-kfold-{fold}-autoencoder.json", "w") as f:
            f.write(json.dumps(config, indent=4))
