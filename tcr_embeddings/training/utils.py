import os
import sys
from tcr_embeddings import runtime_constants

os.chdir(runtime_constants.HOME_PATH)
sys.path.append(str(runtime_constants.HOME_PATH))

import ast
import datetime
import json
import time

from tcr_embeddings.embed.physicochemical import *
from tcr_embeddings.embed.llm import tcrbert
from sceptr import variant
from tcr_embeddings.reduction.reduction import AutoEncoder, JohnsonLindenstarauss, NoReduce

global logger, log_fname, start_time
start_time = time.time()


def printf(msg, severity=""):
    logger.write(msg, severity=severity)


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

    match configs["encoding"]:
        case "atchley":
            configs["encoding-method"] = atchley()

        case "kidera":
            configs["encoding-method"] = kidera()

        case "rand":
            configs["encoding-method"] = rand()

        case "aaprop":
            configs["encoding-method"] = aaprop()

        case "tcrbert":
            configs["encoding-method"] = tcrbert()

        case "sceptr-tiny":
            configs["encoding-method"] = variant.tiny()

        case "sceptr-default":
            configs["encoding-method"] = variant.default()

        case _:
            raise ValueError

    return configs


def load_reduction(config):
    match config["reduction"]:
        case "autoencoder":
            reducer = AutoEncoder(config["encoding-method"], config["encoding"])

        case "johnson-lindenstarauss":
            sample = pd.read_csv(Path.cwd() / "data/sample.tsv", sep="\t", dtype=str)
            in_dim = config["encoding-method"].calc_vector_representations(sample)
            reducer = JohnsonLindenstarauss(in_dim)

        case _:
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
    signature = {"classifier.pth", "eval-records.csv", "test-records.csv", "train-records.csv"}
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
