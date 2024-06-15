import json
from pathlib import Path

dir = Path(__file__).resolve().parent

encodings = ["atchley", "kidera", "rand", "aaprop", "tcrbert", "sceptr-default", "sceptr-tiny"]
kfolds = 5
config_path_base = Path.cwd() / "configs/config.json"

with open(dir / config_path_base, "r") as file:
    config = json.load(file)

for encoding in encodings:
    for fold in range(kfolds):
        fname = f"{encoding}-kfold-{fold}.json"
        config["encoding"] = encoding
        config["kfold"] = fold
        config["output-path"] = f"results/{encoding}/kfold-{fold}"
        config["reduction"] = ""
        with open(dir / fname, "w") as f:
            f.write(json.dumps(config, indent = 4))