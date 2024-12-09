import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tcr_embeddings import constants, runtime_constants
from tcr_embeddings.training import training_utils

os.chdir(runtime_constants.HOME_PATH)
sys.path.append(str(runtime_constants.HOME_PATH))

global start_time, logger

if __name__ == "__main__":
    OUTPUT_PATH: None | Path = None
    classifier: None | torch.nn.Module = None
    try:
        parser = training_utils.parse_args()
        training_utils.check_if_only_create(parser)

        # Up to here, if we continue, parser --make is false.
        config_file = parser.config if parser.config is not None else "config.json"
        with open(config_file, "r") as f:
            configs = training_utils.load_configs(json.load(f))

        # Loading configurations & constants
        OUTPUT_PATH = training_utils.make_directory_where_necessary(
            runtime_constants.HOME_PATH / configs["output-path"]
        )

        logger = training_utils.create_logger(OUTPUT_PATH, parser)

        arg = " ".join(sys.argv)
        training_utils.printf_configs(configs)
        training_utils.printf(
            "Arguments: python " + " ".join(sys.argv), severity="INFO"
        )
        training_utils.printf_cuda_configs()

        np.random.seed(constants.RAND_SEED)
        classifier = training_utils.create_classifier(configs)
        optim = training_utils.create_optimiser(classifier)
        dataset = training_utils.create_dataset(configs)

        # Creating Training Records
        loss_record: dict[str, list[float]] = {"train": [], "val": [], "test": []}
        training_utils.start_timer()
        for e in range(constants.EPOCHS):
            out_path = training_utils.make_directory_where_necessary(
                OUTPUT_PATH / f"Epoch {e}"
            )
            kwargs = {
                "dataset": dataset,
                "classifier": classifier,
                "configs": configs,
                "current_file_no": 0,
                "optim": optim,
                "current_epoch": e,
            }

            training_utils.printf(f"Epoch {e} / {constants.EPOCHS} - Training")
            dataset.train()
            classifier.train()
            records = training_utils.iterate_through_files(**kwargs)  # type: ignore
            pd.DataFrame(records).to_csv(out_path / "train-records.csv", index=False)
            loss_record["train"].append(np.mean(records["loss"]))
            # we expect kwargs to always be integer
            kwargs["current_file_no"] += len(dataset)  # type: ignore

            training_utils.printf(f"Epoch {e} / {constants.EPOCHS} - Validation")
            dataset.validation()
            classifier.eval()
            with torch.no_grad():
                records = training_utils.iterate_through_files(**kwargs)  # type: ignore
            pd.DataFrame(records).to_csv(out_path / "eval-records.csv", index=False)
            loss_record["val"].append(np.mean(records["loss"]))
            # we expect kwargs to always be integer
            kwargs["current_file_no"] += len(dataset)  # type: ignore

            training_utils.printf(
                f"Epoch {e} / {constants.EPOCHS} - Testing (K-Fold CV)."
            )
            dataset.test()
            classifier.eval()
            with torch.no_grad():
                records = training_utils.iterate_through_files(**kwargs)  # type: ignore
            pd.DataFrame(records).to_csv(out_path / "test-records.csv", index=False)
            loss_record["test"].append(np.mean(records["loss"]))
            # we expect kwargs to always be integer
            kwargs["current_file_no"] += len(dataset)  # type: ignore

            training_utils.summarise_epoch(e, loss_record, classifier, out_path)

        training_utils.export_kfold_set(OUTPUT_PATH, configs)

    except KeyboardInterrupt:
        training_utils.printf("Interrupted", "INFO")

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        training_utils.printf("Error Encountered: Logging Information", "ERROR")
        if torch.cuda.is_available():
            training_utils.printf(
                f"Torch Memory Taken: {torch.cuda.memory_allocated()}"
            )

        training_utils.printf(
            (
                f"Line {exc_tb.tb_lineno} - "
                if exc_tb is not None
                else "" + f"{type(e).__name__}: {str(e)}"
            ),
            "ERROR",
        )

    finally:
        if OUTPUT_PATH is not None and classifier is not None:
            torch.save(classifier.state_dict(), OUTPUT_PATH / "classifier-trained.pth")
        logger.close()
