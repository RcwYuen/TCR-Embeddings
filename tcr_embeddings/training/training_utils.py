import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sceptr import Sceptr, variant

from tcr_embeddings import constants, runtime_constants
from tcr_embeddings.embed.llm import LLMEncoder, tcrbert
from tcr_embeddings.embed.physicochemical import (
    PhysicoChemicalEncoder,
    aaprop,
    atchley,
    kidera,
    rand,
)
from tcr_embeddings.reduction.reduction import (
    AutoEncoder,
    JohnsonLindenstarauss,
    NoReduce,
)
from tcr_embeddings.training.dataloader import Patients
from tcr_embeddings.training.logger import Logger
from tcr_embeddings.training.models import (
    MIClassifier,
    ordinary_classifier,
    reduced_classifier,
)

os.chdir(runtime_constants.HOME_PATH)
sys.path.append(str(runtime_constants.HOME_PATH))

logger: None | Logger = None
start_time: None | float = None

"""
TODO:
- Add printing method to all constants (so resume can run based on the previously set constants not the new ones).
- Add check to the end of iteration function, so it makes sure that all gradients are descended before 
  concluding the batch.
- Add interpretability executions to the end of training loop (at the end of all epochs).
"""


def create_logger(output_path, parser) -> Logger:
    global logger
    logger = Logger(
        output_path / parser.log_file
        if parser.log_file is not None
        else output_path / "training.log"
    )
    return logger


def start_timer() -> None:
    global start_time
    start_time = time.time()


def printf(text: str, severity: str = "") -> None:
    if logger is None:
        raise ValueError("Logger not instantiated.")
    logger.write(text, severity=severity)


def printf_configs(custom_configs: dict) -> None:
    for k, v in custom_configs.items():
        printf(f"Config {k}: {v}", severity="INFO")


def printf_cuda_configs():
    if torch.cuda.is_available():
        printf(f"Torch CUDA Device Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            printf(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")


def projected_time(
    current_file_no: int, total_file_no: int, current_epoch: int, total_epoch: int
) -> tuple[str, float]:
    if start_time is None:
        raise ValueError("start_time is not instantiated.")

    elapsed_time = time.time() - start_time
    total_processed_files = current_epoch * total_file_no + current_file_no
    total_files_to_do = total_epoch * total_file_no - total_processed_files
    rate_of_processing = total_processed_files / elapsed_time
    return str(
        datetime.datetime.now()
        + datetime.timedelta(seconds=total_files_to_do / rate_of_processing)
    ), total_processed_files / (total_epoch * total_file_no)


def get_reduction(config: dict) -> JohnsonLindenstarauss | AutoEncoder | NoReduce:
    match config["reduction"]:
        case "autoencoder":
            return AutoEncoder(config["encoding-method"], config["encoding"])

        case "johnson-lindenstarauss":
            if hasattr(config["encoding-method"], "get_out_dim"):
                return JohnsonLindenstarauss(config["encoding-method"].get_out_dim())
            else:
                return JohnsonLindenstarauss(
                    config["encoding-method"]
                    .calc_vector_representations(runtime_constants.DF_SAMPLE)
                    .shape[1]
                )

        case "no-reduction":
            return NoReduce()

        case _:
            raise ValueError("Unrecognised reduction method: " + config["reduction"])


def make_directory_where_necessary(directory: Path) -> Path:
    if not os.path.exists(directory):
        if make_directory_where_necessary(directory.parent):
            os.mkdir(directory)
    return directory


def get_embedding_method(
    encoding_str: str,
) -> PhysicoChemicalEncoder | Sceptr | LLMEncoder:
    match encoding_str:
        case "atchley":
            return atchley()

        case "kidera":
            return kidera()

        case "rand":
            return rand()

        case "aaprop":
            return aaprop()

        case "tcrbert":
            return tcrbert()

        case "sceptr-tiny":
            return variant.tiny()

        case "sceptr-default":
            return variant.default()

        case _:
            raise ValueError("Unrecognised encoding method.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Location for Configuration File")
    parser.add_argument(
        "-m", "--make", action="store_true", help="Create Default Configuration JSON"
    )
    parser.add_argument("--dir", help="Directory for Training to be resumed.")
    parser.add_argument("-l", "--log-file", help="File Logger Name")
    return parser.parse_args()


def load_default_configs(write: bool = False) -> dict:
    configs = {
        "output-path": "results",
        "kfold": 0,
        "encoding": "atchley",
        "reduction": "no-reduction",
    }

    if write:
        with open("config.json", "w") as outs:
            outs.write(json.dumps(configs, indent=4))

    return configs


def load_configs(custom_configs: dict) -> dict:
    configs = load_default_configs(write=False)
    for key, val in custom_configs.items():
        if key in configs:
            configs[key] = val
        else:
            raise ValueError(f"Unrecognised Configuration Found: {key}")

    configs["encoding-method"] = get_embedding_method(configs["encoding"])
    configs["reducer"] = get_reduction(configs)
    return configs


def check_if_only_create(parser: argparse.Namespace) -> None:
    if parser.make:
        print("Creating Configuration Template")
        load_default_configs(write=True)
        quit()


def calculate_embeddings(df: pd.DataFrame, configs: dict) -> torch.Tensor:
    embeddings = configs["encoding-method"].calc_vector_representations(df)
    embeddings = configs["reducer"].reduce(embeddings)
    embeddings = torch.from_numpy(embeddings).to(torch.float32)
    return (
        embeddings.cuda()
        if torch.cuda.is_available() and constants.USE_CUDA
        else embeddings
    )


def create_label(label: int, pred: torch.Tensor) -> torch.Tensor:
    label_t = torch.full_like(pred, label, dtype=torch.float32)
    return (
        label_t.cuda() if torch.cuda.is_available() and constants.USE_CUDA else label_t
    )


def create_optimiser(classifier: torch.nn.Module) -> torch.optim.Optimizer:
    optim = torch.optim.Adam(
        classifier.parameters(), lr=constants.LR, weight_decay=constants.L2_PENALTY
    )
    optim.zero_grad()
    return optim


def create_classifier(configs: dict) -> MIClassifier:
    if configs["reduction"] == "no-reduction":
        return ordinary_classifier(configs["encoding-method"])
    else:
        return reduced_classifier()


def create_dataset(configs: dict) -> Patients:
    dataset = Patients(
        split=constants.TRAIN_TEST_SPLIT,
        positives=constants.PATH_POSITIVE_CLASS,
        negatives=constants.PATH_NEGATIVE_CLASS,
        kfold=configs["kfold"],
    )

    for k, v in dataset.files_by_category().items():
        printf(f"{k} Data: {v} Entries", severity="INFO")

    printf("Data loaded.  Commencing Training.")
    return dataset


def iterate_through_files(
    dataset: Patients,
    classifier: MIClassifier,
    configs: dict,
    current_file_no: int,
    optim: torch.optim.Optimizer,
    current_epoch: int,
) -> dict:
    current_epoch_records: dict[str, list[float]] = {
        "pred": [],
        "actual": [],
        "seqs": [],
        "loss": [],
    }
    # we validate whether dataset mode is same as gradients
    if (dataset.get_mode() == "train") ^ torch.is_grad_enabled():
        raise ValueError("Dataset mode and gradient mode does not match.")

    label: int
    df: pd.DataFrame
    idx: int

    # type ignore as dataset inherits torch.utils.data.Dataset, which is an iterable
    for idx, (label, df) in enumerate(dataset):  # type: ignore
        printf(f"Processing File {idx} / {len(dataset)}.  True Label {label}")
        pred = classifier(calculate_embeddings(df, configs))
        loss = runtime_constants.CRITERION(pred, create_label(label, pred))

        current_epoch_records["pred"].append(pred.item())
        current_epoch_records["actual"].append(label)
        current_epoch_records["seqs"].append(len(df))
        current_epoch_records["loss"].append(loss.item())
        printf(f"File {idx} / {len(dataset)}: Predicted Value: {pred.item()}")
        printf(f"File {idx} / {len(dataset)}: Loss: {loss.item()}")

        if torch.is_grad_enabled():
            # scaling loss based on class imbalance
            loss /= dataset.ratio(label)
            loss.backward()

            if ((idx + 1) % constants.ACCUMMULATION == 0) or (idx == len(dataset) - 1):
                printf("Updating Network")
                ls_accummulated_loss = current_epoch_records["loss"][
                    -constants.ACCUMMULATION :
                ]
                printf(
                    f"Accummulated Losses (Averaged): {np.mean(ls_accummulated_loss)}"
                )
                optim.step()
                optim.zero_grad()

        completion_time, pctdone = projected_time(
            current_file_no + idx + 1,
            dataset.total_files(),
            current_epoch,
            constants.EPOCHS,
        )
        printf(f"Projected Completion Date: {completion_time}")
        printf(f"Percentage Done: {round(pctdone * 100, 5)}%")

    return current_epoch_records


def summarise_epoch(
    current_epoch: int,
    loss_record: dict,
    classifier: torch.nn.Module,
    output_path: Path,
) -> None:
    printf(f"Epoch {current_epoch} / {constants.EPOCHS} - Completed.")
    printf(
        f"Epoch {current_epoch} / {constants.EPOCHS} - Average Training Loss: "
        + loss_record["train"][-1]
    )
    printf(
        f"Epoch {current_epoch} / {constants.EPOCHS} - Average Validation Loss: "
        + loss_record["val"][-1]
    )
    printf(
        f"Epoch {current_epoch} / {constants.EPOCHS} - Average Testing Loss: "
        + loss_record["test"][-1]
    )
    printf("Saving Model Checkpoint.")
    torch.save(classifier.state_dict(), output_path / "classifier.pth")


def export_kfold_set(pth: Path, configs: dict) -> None:
    for i, pos_path in configs["positive-path"]:
        with open(pth / "positive-kfold-set.txt", "w") as outfile:
            with open(
                runtime_constants.HOME_PATH / pos_path / "kfold.txt", "r"
            ) as positive_kfold:
                outfile.writelines(positive_kfold.readlines()[configs["kfold"]] + "\n")

    for i, neg_path in configs["negative-path"]:
        with open(pth / "negative-kfold.txt", "w") as outfile:
            with open(
                runtime_constants.HOME_PATH / neg_path / "kfold.txt", "r"
            ) as negative_kfold:
                outfile.writelines(negative_kfold.readlines()[configs["kfold"]] + "\n")
