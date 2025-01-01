import json
from pathlib import Path

from tcr_embeddings.training.training_utils import (
    load_default_configs,
    make_directory_where_necessary,
)

loc = Path(__file__).resolve().parent

ls_encodings = ["atchley", "kidera", "rand", "aaprop", "sceptr-default", "sceptr-tiny"]
ls_reduction_methods = ["johnson-lindenstarauss", "autoencoder", "no-reduction"]
kfolds = 5

if __name__ == "__main__":
    config = load_default_configs()

    for encoding in ls_encodings:
        for fold in range(kfolds):
            for reduction in ls_reduction_methods:
                store_location = make_directory_where_necessary(
                    loc / f"{encoding}/{reduction}"
                )
                config["encoding"] = encoding
                config["kfold"] = fold
                config["output-path"] = f"results/{encoding}-{reduction}/kfold-{fold}"
                config["reduction"] = reduction
                config["use-pre-embedded-if-possible"] = True

                with open(store_location / f"kfold-{fold}.json", "w") as f:
                    f.write(json.dumps(config, indent=4))
