import copy
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tcr_embeddings import runtime_constants
from tcr_embeddings.interpret import interpretability
from tcr_embeddings.training import training_utils as utils

os.chdir(runtime_constants.HOME_PATH)
sys.path.append(str(runtime_constants.HOME_PATH))

global start_time, logger

if __name__ == "__main__":
    OUTPUT_PATH: None | Path = None
    classifier: None | torch.nn.Module = None
    try:
        parser = utils.parse_args()
        utils.check_if_only_create(parser)

        # Up to here, if we continue, parser --make is false.
        config_file = parser.config if parser.config is not None else "config.json"
        with open(config_file, "r") as f:
            configs = utils.load_configs(json.load(f))

        # Loading configurations & runtime_constants
        OUTPUT_PATH = utils.make_directory_where_necessary(
            runtime_constants.HOME_PATH / configs["output-path"]
        )

        logger = utils.create_logger(OUTPUT_PATH, parser.log_file)

        utils.printf_configs(configs)
        utils.printf("Arguments: python " + " ".join(sys.argv), severity="INFO")
        utils.printf_cuda_configs()

        np.random.seed(runtime_constants.RAND_SEED)
        classifier = utils.create_classifier(configs)
        optim = utils.create_optimiser(classifier)
        dataset = utils.create_dataset(configs)
        utils.output_extension_type(dataset)

        # Creating Training Records
        loss_record: dict[str, list[float]] = {"train": [], "val": [], "test": []}
        utils.start_timer()
        for e in range(runtime_constants.EPOCHS):
            out_path = utils.make_directory_where_necessary(OUTPUT_PATH / f"Epoch {e}")
            kwargs = {
                "dataset": dataset,
                "classifier": classifier,
                "configs": configs,
                "current_file_no": 0,
                "optim": optim,
                "current_epoch": e,
            }

            utils.printf(f"Epoch {e} / {runtime_constants.EPOCHS} - Training")
            dataset.train()
            classifier.train()
            records = utils.iterate_through_files(**kwargs)  # type: ignore
            pd.DataFrame(records).to_csv(out_path / "train-records.csv", index=False)
            loss_record["train"].append(
                np.mean(records["loss"]) if len(records["loss"]) != 0 else -1
            )
            # we expect kwargs to always be integer
            kwargs["current_file_no"] += len(dataset)  # type: ignore

            utils.printf(f"Epoch {e} / {runtime_constants.EPOCHS} - Validation")
            dataset.validation()
            classifier.eval()
            with torch.no_grad():
                records = utils.iterate_through_files(**kwargs)  # type: ignore
            pd.DataFrame(records).to_csv(out_path / "eval-records.csv", index=False)
            loss_record["val"].append(
                np.mean(records["loss"]) if len(records["loss"]) != 0 else -1
            )
            # we expect kwargs to always be integer
            kwargs["current_file_no"] += len(dataset)  # type: ignore

            utils.printf(
                f"Epoch {e} / {runtime_constants.EPOCHS} - Testing (K-Fold CV)."
            )
            dataset.test()
            classifier.eval()
            with torch.no_grad():
                records = utils.iterate_through_files(**kwargs)  # type: ignore
            pd.DataFrame(records).to_csv(out_path / "test-records.csv", index=False)
            loss_record["test"].append(
                np.mean(records["loss"]) if len(records["loss"]) != 0 else -1
            )
            # we expect kwargs to always be integer
            kwargs["current_file_no"] += len(dataset)  # type: ignore

            utils.summarise_epoch(e, loss_record, classifier, out_path)

        utils.export_kfold_set(OUTPUT_PATH, configs)
        utils.export_constants(OUTPUT_PATH)

        utils.printf("Finding interpretability scripts", "INFO")
        interpretability.run(
            configs=configs,
            dataset=dataset,
            model=copy.deepcopy(classifier),
            outpath=OUTPUT_PATH,
        )

        utils.printf("All Done, cleaning up", "INFO")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        utils.printf("Interrupted", "INFO")

    except Exception as e:
        utils.printf_exceptions_raised(e)

    finally:
        if OUTPUT_PATH is not None and classifier is not None:
            torch.save(classifier.state_dict(), OUTPUT_PATH / "classifier-trained.pth")
        utils.close_logger()
