import os
import sys
from pathlib import Path

from dotenv import load_dotenv

dir = Path(__file__).resolve().parent
load_dotenv(Path.cwd() / ".env")
python_path = os.getenv("PYTHONPATH")
if python_path:
    sys.path.append(python_path)

import argparse
import ast
import datetime
import json
import time

import numpy as np
import pandas as pd
import torch

from training.dataloader import Patients
from training.logger import Logger
from training.models import ordinary_classifier, reduced_classifier


def printf(msg, severity=""):
    logger.write(msg, severity=severity)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Directory for Training to be resumed.")
    return parser.parse_args()


def defaults(write=False):
    configs = {
        "positive-path": ["data"],
        "negative-path": ["data"],
        "output-path": "results",
        "epoch": 100,
        "lr": 1e-3,
        "kfold": 0,
        "train-split": 0.8,
        "accummulation": 4,
        "l2-penalty": 0,
        "encoding": "atchley",
        "reduction": "johnson-lindenstarauss",
        "seed": 42,
    }

    if write:
        with open("config.json", "w") as outs:
            outs.write(json.dumps(configs, indent=4))

    return configs


def load_configs():
    logfile = open(log_fname, "r")
    finished_loading = False
    configs = {}

    while not finished_loading:
        newline = logfile.readline().replace("\n", "")
        if "Config" in newline:
            cf, arg = newline.split(" Config ")[-1].split(": ")
            if arg == "":
                configs[cf] = arg
            else:
                try:
                    configs[cf] = ast.literal_eval(arg)
                except ValueError:
                    configs[cf] = arg
                except SyntaxError:
                    pass
        else:
            finished_loading = True

    if configs["encoding"] == "atchley":
        from embed.physicochemical import atchley

        configs["encoding-method"] = atchley()

    elif configs["encoding"] == "kidera":
        from embed.physicochemical import kidera

        configs["encoding-method"] = kidera()

    elif configs["encoding"] == "rand":
        from embed.physicochemical import rand

        configs["encoding-method"] = rand()

    elif configs["encoding"] == "aaprop":
        from embed.physicochemical import aaprop

        configs["encoding-method"] = aaprop()

    elif configs["encoding"] == "tcrbert":
        from embed.llm import tcrbert

        configs["encoding-method"] = tcrbert()

    elif configs["encoding"] == "sceptr-tiny":
        from sceptr import variant

        configs["encoding-method"] = variant.tiny()

    elif configs["encoding"] == "sceptr-default":
        from sceptr import variant

        configs["encoding-method"] = variant.default()

    return configs


def load_reduction(config):
    if config["reduction"] == "autoencoder":
        from reduction.reduction import AutoEncoder

        reducer = AutoEncoder(config["encoding-method"], config["encoding"])

    elif config["reduction"] == "johnson-lindenstarauss":
        from reduction.reduction import JohnsonLindenstarauss

        sample = pd.read_csv(Path.cwd() / "data/sample.tsv", sep="\t", dtype=str)
        in_dim = config["encoding-method"].calc_vector_representations(sample)
        reducer = JohnsonLindenstarauss(in_dim)

    else:
        from reduction.reduction import NoReduce

        config["reduction"] = None
        reducer = NoReduce()

    return reducer


def make_directory_where_necessary(directory):
    if not os.path.exists(directory):
        if make_directory_where_necessary(directory.parent):
            os.mkdir(directory)
    return directory


def load_last_epoch(custom_configs):
    loc = Path.cwd() / custom_configs["output-path"]
    maxepoch = max([int(i.name.replace("Epoch ", "")) for i in loc.glob("Epoch */")])
    signature = set(
        ["classifier.pth", "eval-records.csv", "test-records.csv", "train-records.csv"]
    )
    exists = set([i for i in (loc / f"Epoch {maxepoch}").glob("*") if i in signature])
    return maxepoch - 1 if signature - exists else maxepoch


def projected_time(current_file_no, total_file_no, current_epoch, total_epoch):
    elapsed_time = time.time() - start_time
    total_processed_files = current_epoch * total_file_no + current_file_no
    total_files_to_do = total_epoch * total_file_no - total_processed_files
    rate_of_processing = total_processed_files / elapsed_time
    return str(
        datetime.datetime.now()
        + datetime.timedelta(seconds=total_files_to_do / rate_of_processing)
    ), total_processed_files / (total_epoch * total_file_no)


global logger, log_fname

if __name__ == "__main__":
    try:
        parser = parse_args()
        outpath = parser.dir
        log_dir = list((Path.cwd() / outpath).glob("*.log"))
        assert len(log_dir) == 1, "Ambiguous Log Files"
        log_fname = log_dir[0]
        logger = Logger(log_fname, opening_mode="a+")
        arg = " ".join(sys.argv)
        printf("Arguments: python " + " ".join(sys.argv), severity="INFO")

        if torch.cuda.is_available():
            printf(f"Torch CUDA Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                printf(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")

        printf("Resuming Training - Loading Configurations")
        custom_configs = load_configs()
        custom_configs["reducer"] = load_reduction(custom_configs)
        custom_configs["output-path"] = outpath
        np.random.seed(custom_configs["seed"])
        printf("Configurations Loaded.")

        # Loading Classifier
        if custom_configs["reduction"] is None:
            classifier = ordinary_classifier(
                custom_configs["encoding-method"], custom_configs["use-cuda"]
            )
        else:
            classifier = reduced_classifier(custom_configs["use-cuda"])

        last_epoch = load_last_epoch(custom_configs)
        classifier.load_state_dict(
            torch.load(
                Path.cwd()
                / custom_configs["output-path"]
                / f"Epoch {last_epoch}"
                / "classifier.pth",
                map_location="cuda" if torch.cuda.is_available() else "cpu",
            )
        )

        optim = torch.optim.Adam(
            classifier.parameters(),
            lr=custom_configs["lr"],
            weight_decay=custom_configs["l2-penalty"],
        )
        criterion = torch.nn.BCELoss()
        optim.zero_grad()

        # Loading Data
        dataset = Patients(
            split=custom_configs["train-split"],
            positives=custom_configs["positive-path"],
            negatives=custom_configs["negative-path"],
            kfold=custom_configs["kfold"],
        )

        for k, v in dataset.files_by_category().items():
            printf(f"{k} Data: {v} Entries", severity="INFO")

        # Creating Training Records
        loss_record = {"train": [], "val": [], "test": []}
        total_epochs = custom_configs["epoch"]

        start_time = time.time()
        # Start of Training Loop
        for e in range(last_epoch + 1, total_epochs, 1):
            current_epoch_records = {"pred": [], "actual": [], "seqs": [], "loss": []}
            outpath = make_directory_where_necessary(
                Path.cwd() / custom_configs["output-path"] / f"Epoch {e}"
            )

            printf(f"Epoch {e} / {total_epochs} - Training")
            dataset.train()
            classifier.train()
            current_file_no = 0

            for idx, train_samples in enumerate(dataset):
                current_file_no += 1
                label, df = train_samples

                printf(f"Processing File {idx} / {len(dataset)}.  True Label {label}")

                # Embedding File
                embeddings = custom_configs[
                    "encoding-method"
                ].calc_vector_representations(df)

                # Reducing Dimensionality
                embeddings = custom_configs["reducer"].reduce(embeddings)

                # Tensoring
                embeddings = torch.from_numpy(embeddings).to(torch.float32)
                embeddings = (
                    embeddings.cuda()
                    if torch.cuda.is_available() and custom_configs["use-cuda"]
                    else embeddings
                )

                # Creating Prediction
                prediction = classifier(embeddings)
                label_t = torch.full_like(prediction, label, dtype=torch.float32)
                label_t = (
                    label_t.cuda()
                    if torch.cuda.is_available() and custom_configs["use-cuda"]
                    else label_t
                )
                current_epoch_records["pred"].append(prediction.data.tolist()[0][0])
                current_epoch_records["actual"].append(label)
                current_epoch_records["seqs"].append(len(df))

                # Computing Gradients
                loss = criterion(prediction, label_t)
                current_epoch_records["loss"].append(loss.data.tolist())
                printf(
                    f"File {idx} / {len(dataset)}: Predicted Value: {prediction.data.tolist()[0][0]}"
                )
                printf(f"File {idx} / {len(dataset)}: Loss: {loss.data.tolist()}")

                loss /= dataset.ratio(label)
                loss.backward()

                if (idx + 1) % custom_configs["accummulation"] == 0:
                    printf("Updating Network")
                    accummulated = current_epoch_records["loss"][
                        -custom_configs["accummulation"] :
                    ]
                    printf(
                        f"Accummulated Losses (Unnormalised): {str([round(i, 4) for i in accummulated])}"
                    )
                    printf(f"Accummulated Losses (Averaged): {np.mean(accummulated)}")
                    optim.step()
                    optim.zero_grad()

                completion_time, pctdone = projected_time(
                    current_file_no, dataset.total_files(), e, total_epochs
                )
                printf(f"Projected Completion Date: {completion_time}")
                printf(f"Percentage Done: {round(pctdone * 100, 5)}%")

            pd.DataFrame(current_epoch_records).to_csv(
                outpath / "train-records.csv", index=False
            )
            loss_record["train"].append(np.mean(current_epoch_records["loss"]))
            current_epoch_records = {"pred": [], "actual": [], "seqs": [], "loss": []}

            printf(f"Epoch {e} / {total_epochs} - Validation")
            dataset.validation()
            classifier.eval()

            with torch.no_grad():
                for idx, train_samples in enumerate(dataset):
                    current_file_no += 1
                    label, df = train_samples

                    printf(
                        f"Processing File {idx} / {len(dataset)}.  True Label {label}"
                    )
                    embeddings = custom_configs[
                        "encoding-method"
                    ].calc_vector_representations(df)
                    embeddings = custom_configs["reducer"].reduce(embeddings)
                    embeddings = torch.from_numpy(embeddings).to(torch.float32)
                    embeddings = (
                        embeddings.cuda()
                        if torch.cuda.is_available() and custom_configs["use-cuda"]
                        else embeddings
                    )
                    prediction = classifier(embeddings)
                    label_t = torch.full_like(prediction, label, dtype=torch.float32)
                    label_t = (
                        label_t.cuda()
                        if torch.cuda.is_available() and custom_configs["use-cuda"]
                        else label_t
                    )
                    current_epoch_records["pred"].append(prediction.data.tolist()[0][0])
                    current_epoch_records["actual"].append(label)
                    current_epoch_records["seqs"].append(len(df))
                    loss = criterion(prediction, label_t)
                    current_epoch_records["loss"].append(loss.data.tolist())

                    printf(
                        f"File {idx} / {len(dataset)}: Predicted Value: {prediction.data.tolist()[0][0]}"
                    )
                    printf(f"File {idx} / {len(dataset)}: Loss: {loss.data.tolist()}")
                    completion_time, pctdone = projected_time(
                        current_file_no, dataset.total_files(), e, total_epochs
                    )
                    printf(f"Projected Completion Date: {completion_time}")
                    printf(f"Percentage Done: {round(pctdone * 100, 5)}%")

            pd.DataFrame(current_epoch_records).to_csv(
                outpath / "eval-records.csv", index=False
            )
            loss_record["val"].append(np.mean(current_epoch_records["loss"]))
            current_epoch_records = {"pred": [], "actual": [], "seqs": [], "loss": []}

            printf(
                f"Epoch {e} / {total_epochs} - Testing.  Note: This is for K-Fold CV."
            )
            dataset.test()
            classifier.eval()

            with torch.no_grad():
                for idx, train_samples in enumerate(dataset):
                    current_file_no += 1
                    label, df = train_samples

                    printf(
                        f"Processing File {idx} / {len(dataset)}.  True Label {label}"
                    )
                    embeddings = custom_configs[
                        "encoding-method"
                    ].calc_vector_representations(df)
                    embeddings = custom_configs["reducer"].reduce(embeddings)
                    embeddings = torch.from_numpy(embeddings).to(torch.float32)
                    embeddings = (
                        embeddings.cuda()
                        if torch.cuda.is_available() and custom_configs["use-cuda"]
                        else embeddings
                    )
                    prediction = classifier(embeddings)
                    label_t = torch.full_like(prediction, label, dtype=torch.float32)
                    label_t = (
                        label_t.cuda()
                        if torch.cuda.is_available() and custom_configs["use-cuda"]
                        else label_t
                    )
                    current_epoch_records["pred"].append(prediction.data.tolist()[0][0])
                    current_epoch_records["actual"].append(label)
                    current_epoch_records["seqs"].append(len(df))
                    loss = criterion(prediction, label_t)
                    current_epoch_records["loss"].append(loss.data.tolist())

                    printf(
                        f"File {idx} / {len(dataset)}: Predicted Value: {prediction.data.tolist()[0][0]}"
                    )
                    printf(f"File {idx} / {len(dataset)}: Loss: {loss.data.tolist()}")
                    completion_time, pctdone = projected_time(
                        current_file_no, dataset.total_files(), e, total_epochs
                    )
                    printf(f"Projected Completion Date: {completion_time}")
                    printf(f"Percentage Done: {round(pctdone * 100, 5)}%")

            pd.DataFrame(current_epoch_records).to_csv(
                outpath / "test-records.csv", index=False
            )
            loss_record["test"].append(np.mean(current_epoch_records["loss"]))

            printf(f"Epoch {e} / {total_epochs} - Completed.")
            printf(
                "Epoch {} / {} - Average Training Loss:   {}".format(
                    e, total_epochs, loss_record["train"][0]
                )
            )
            printf(
                "Epoch {} / {} - Average Validation Loss: {}".format(
                    e, total_epochs, loss_record["val"][0]
                )
            )
            printf("Saving Model Checkpoint.")
            torch.save(classifier.state_dict(), outpath / "classifier.pth")

    except KeyboardInterrupt:
        printf("Interrupted", "INFO")

    finally:
        try:
            torch.save(
                classifier.state_dict(),
                Path.cwd() / custom_configs["output-path"] / "classifier-trained.pth",
            )
        except NameError:
            print("Cannot Save Final Classifier")
