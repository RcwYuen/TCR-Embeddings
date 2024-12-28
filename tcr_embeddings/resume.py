import copy
import gc
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


global logger, log_fname

if __name__ == "__main__":
    OUTPUT_PATH: None | Path = None
    classifier: None | torch.nn.Module = None

    try:
        OUTPUT_PATH = Path.cwd() / utils.parse_args().dir
        assert (
            isinstance(OUTPUT_PATH, Path)
            and len(log_dir := list(OUTPUT_PATH.glob("*.log"))) == 1
        ), "Ambiguous Log Files"
        logger = utils.create_logger(OUTPUT_PATH, log_dir[0].name, opening_mode="a+")
        configs = utils.find_configs_from_log(log_dir[0])
        last_epoch = utils.load_last_epoch(configs)

        utils.printf("Arguments: python " + " ".join(sys.argv), severity="INFO")
        utils.printf_cuda_configs()
        utils.printf("Resuming Training - Loading Configurations")

        np.random.seed(runtime_constants.RAND_SEED)
        classifier = utils.create_classifier(
            configs,
            trained_classifier_pth=OUTPUT_PATH / f"Epoch {last_epoch}/classifier.pth",
        )

        optim = utils.create_optimiser(classifier)
        dataset = utils.create_dataset(configs)

        # Creating Training Records
        loss_record: dict[str, list[float]] = {"train": [], "val": [], "test": []}
        utils.start_timer()
        # Start of Training Loop
        for e in range(last_epoch + 1, runtime_constants.EPOCHS, 1):
            outpath = utils.make_directory_where_necessary(OUTPUT_PATH / f"Epoch {e}")
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
            pd.DataFrame(records).to_csv(outpath / "train-records.csv", index=False)
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
            pd.DataFrame(records).to_csv(outpath / "eval-records.csv", index=False)
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
            pd.DataFrame(records).to_csv(outpath / "test-records.csv", index=False)
            loss_record["test"].append(
                np.mean(records["loss"]) if len(records["loss"]) != 0 else -1
            )
            # we expect kwargs to always be integer
            kwargs["current_file_no"] += len(dataset)  # type: ignore

            utils.summarise_epoch(e, loss_record, classifier, outpath)

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
