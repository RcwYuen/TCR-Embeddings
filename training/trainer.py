from pathlib import Path
from dotenv import load_dotenv
import os, sys

dir = Path(__file__).resolve().parent
load_dotenv(Path.cwd() / ".env")
python_path = os.getenv('PYTHONPATH')
if python_path:
    sys.path.append(python_path)

from training.logger import Logger
from training.models import ordinary_classifier, reduced_classifier
from training.dataloader import Patients
import argparse
import json
import torch
import pandas as pd
import numpy as np
import time, datetime

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        help = "Location for Configuration File"
    )
    parser.add_argument(
        "-m", "--make", action = "store_true",
        help = "Create Default Configuration JSON"
    )
    parser.add_argument(
        "-l", "--log-file", help = "File Logger Name"
    )
    return parser.parse_args()

def defaults(write = False):
    configs = {
        "positive-path": ["data"],
        "negative-path": ["data"],
        "output-path": "training",
        "epoch": 25,
        "lr": 1e-3,
        "kfold": 0,
        "train-split": 0.8,
        "accummulation": 4,
        "l2-penalty": 0,
        "encoding": "atchley",
        "reduction": "johnson-lindenstarauss",
        "seed": 42
    }

    if write:
        with open("config.json", "w") as outs:
            outs.write(json.dumps(configs))

    return configs

def load_configs(custom_configs):
    configs = defaults(write = False)
    for key, val in custom_configs.items():
        print(f"Config: {key}: {val}", severity = "INFO")
        if key in configs:
            configs[key] = val
        else:
            print(
                f"Unrecognised Configuration Found.  Please regenerate the configuration file with 'python {arg} --make'"
            )
            raise ValueError(f"Unrecongised Configuration Found: {key}")
    
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
    
def make_directory_where_necessary(directory):
    if not os.path.exists(directory):
        if make_directory_where_necessary(directory.parent):
            os.mkdir(directory)
    return directory

def load_reduction(config):
    if config["reduction"] == "autoencoder":
        from reduction.reduction import AutoEncoder
        reducer = AutoEncoder(config["encoding-method"], config["encoding"])
    
    elif config["reduction"] == "johnson-lindenstarauss":
        from reduction.reduction import JohnsonLindenstarauss
        sample = pd.read_csv(Path.cwd() / "data/sample.tsv", sep = "\t", dtype = str)
        in_dim = config["encoding-method"].calc_vector_representations(sample)
        reducer = JohnsonLindenstarauss(in_dim)
    
    else:
        from reduction.reduction import NoReduce
        config["reduction"] = None
        reducer = NoReduce()

    return reducer

def projected_time(current_file_no, total_file_no, current_epoch, total_epoch):
    elasped_time = time.time() - start_time
    total_processed_files = current_epoch * total_file_no + current_file_no
    total_files = total_epoch * total_file_no
    seconds_needed = time.time() + elasped_time / total_processed_files * total_files
    return str(datetime.datetime.now() + datetime.timedelta(seconds = seconds_needed))

global start_time

if __name__ == "__main__":
    try:
        parser = parse_args()
        sys.stdout = Logger(parser.log_file if parser.log_file is not None else "training.log")
        
        print("Instanciating")
        arg = " ".join(sys.argv)
        print("Arguments: python " + " ".join(sys.argv), "INFO", silent=True)

    
        if parser.make:
            print("Creating Configuration Template")
            defaults(write_to=True)
            quit()
                
        config_file = (parser.config_file if parser.config_file is not None else "config.json")

        if torch.cuda.is_available():
            print(f"Torch CUDA Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")

        # Loading Configurations
        custom_configs = defaults(write_to=False) 
        custom_configs = load_configs(json.load(open(config_file, "r")))

        # Loading Reducer
        custom_configs["reducer"] = load_reduction(custom_configs)

        # Seeding
        np.random.seed(custom_configs["seed"])
        print ("Random Seeding Completed.")

        # Loading Classifier
        if custom_configs["reduction"] is None:
            classifier = ordinary_classifier(custom_configs["encoding-method"])
        else:
            classifier = reduced_classifier()

        optim = torch.optim.Adam(
            classifier.parameters(),
            lr = custom_configs["lr"],
            weight_decay = custom_configs["l2-penalty"]
        )
        criterion = torch.nn.BCELoss()
        optim.zero_grad()
        print ("Optimiser and Loss Function Instanciated.")

        # Loading Data
        dataset = Patients(
            split = custom_configs["train-split"],
            positives = custom_configs["positive-path"],
            negatives = custom_configs["negative-path"],
            kfold = custom_configs["kfold"]
        )

        print ("Data loaded.  Commencing Training.")

        # Creating Training Records
        loss_record = {"train": [], "val": [], "test": []}
        total_epochs = custom_configs["epoch"]

        start_time = time.time()
        # Start of Training Loop
        for e in range(total_epochs):
            current_epoch_records = {"pred": [], "actual": [], "seqs": [], "loss": []}
            outpath = make_directory_where_necessary(Path.cwd() / custom_configs["output-path"] / f"Epoch {e}")

            print (f"Epoch {e} / {total_epochs} - Training")
            dataset.train()
            classifier.train()
            current_file_no = 0

            for idx, train_samples in enumerate(dataset):
                current_file_no += 1
                label, df = train_samples
                
                print(f"Processing File {idx} / {len(dataset)}.  True Label {label}")
                print("Embedding File")
                embeddings = custom_configs["encoding-method"].calc_vector_representations(df)
                
                print("Reducing Dimensionality")
                embeddings = custom_configs["reducer"].reduce(embeddings)
                
                print("Tensoring")
                embeddings = torch.from_numpy(embeddings).to(torch.float32)
                embeddings = embeddings.cuda() if torch.cuda.is_available() else embeddings
                label_t    = torch.full_like(prediction, label, dtype = torch.float32)
                label_t    = label_t.cuda() if torch.cuda.is_available() else label_t
                
                print("Creating Prediction")
                prediction = classifier(embeddings)
                current_epoch_records["pred"].append(prediction.data.tolist()[0][0])
                current_epoch_records["actual"].append(label)
                current_epoch_records["seqs"].append(len(df))

                print ("Computing Loss & Backpropagation")
                loss = criterion(prediction, label_t)
                current_epoch_records["loss"].append(loss.data.tolist())
                loss /= dataset.ratio(label)
                loss.backward()
                
                print(f"File {i} / {len(dataset)}: Predicted Value: {prediction.data.tolist()[0][0]} ; True Value: {label}")
                print(f"File {i} / {len(dataset)}: Loss: {loss.data.tolist()}")

                if (i + 1) % custom_configs["accummulation"] == 0:
                    print ("Updating Network")
                    accummulated = current_epoch_records["loss"][-custom_configs["accummulation"]:]
                    print (f"Accummulated Losses (Unnormalised): {str([round(i, 4) for i in accummulated])}")
                    print (f"Accummulated Losses (Averaged): {np.mean(accummulated)}")
                    optim.step()
                    optim.zero_grad()
                
                print(f"Projected Completion Date: {projected_time(current_file_no, dataset.total_files(), e, total_epochs)}")
                
            pd.DataFrame(current_epoch_records).to_csv(outpath / "train-records.csv", index = False)
            loss_record["train"].append(np.mean(current_epoch_records["loss"]))
            current_epoch_records = {"pred": [], "actual": [], "seqs": [], "loss": []}

            print (f"Epoch {e} / {total_epochs} - Validation")
            dataset.validation()
            classifier.eval()

            with torch.no_grad():
                for idx, train_samples in enumerate(dataset):
                    current_file_no += 1
                    label, df = train_samples
                    
                    print(f"Processing File {idx} / {len(dataset)}.  True Label {label}")
                    print("Embedding File")
                    embeddings = custom_configs["encoding-method"].calc_vector_representations(df)
                    
                    print("Reducing Dimensionality")
                    embeddings = custom_configs["reducer"].reduce(embeddings)
                    
                    print("Tensoring")
                    embeddings = torch.from_numpy(embeddings).to(torch.float32)
                    embeddings = embeddings.cuda() if torch.cuda.is_available() else embeddings
                    label_t    = torch.full_like(prediction, label, dtype = torch.float32)
                    label_t    = label_t.cuda() if torch.cuda.is_available() else label_t
                    
                    print("Creating Prediction")
                    prediction = classifier(embeddings)
                    current_epoch_records["pred"].append(prediction.data.tolist()[0][0])
                    current_epoch_records["actual"].append(label)
                    current_epoch_records["seqs"].append(len(df))

                    print ("Computing Loss & Backpropagation")
                    loss = criterion(prediction, label_t)
                    current_epoch_records["loss"].append(loss.data.tolist())
                    
                    print(f"File {i} / {len(dataset)}: Predicted Value: {prediction.data.tolist()[0][0]} ; True Value: {label}")
                    print(f"File {i} / {len(dataset)}: Loss: {loss.data.tolist()}")
                    print(f"Projected Completion Date: {projected_time(current_file_no, dataset.total_files(), e, total_epochs)}")

            pd.DataFrame(current_epoch_records).to_csv(outpath / "eval-records.csv", index = False)
            loss_record["val"].append(np.mean(current_epoch_records["loss"]))
            current_epoch_records = {"pred": [], "actual": [], "seqs": [], "loss": []}

            print (f"Epoch {e} / {total_epochs} - Testing.  Note: This is for K-Fold CV.")
            dataset.test()
            classifier.eval()
            
            with torch.no_grad():
                for idx, train_samples in enumerate(dataset):
                    current_file_no += 1
                    label, df = train_samples
                    
                    print(f"Processing File {idx} / {len(dataset)}.  True Label {label}")
                    print("Embedding File")
                    embeddings = custom_configs["encoding-method"].calc_vector_representations(df)
                    
                    print("Reducing Dimensionality")
                    embeddings = custom_configs["reducer"].reduce(embeddings)
                    
                    print("Tensoring")
                    embeddings = torch.from_numpy(embeddings).to(torch.float32)
                    embeddings = embeddings.cuda() if torch.cuda.is_available() else embeddings
                    label_t    = torch.full_like(prediction, label, dtype = torch.float32)
                    label_t    = label_t.cuda() if torch.cuda.is_available() else label_t
                    
                    print("Creating Prediction")
                    prediction = classifier(embeddings)
                    current_epoch_records["pred"].append(prediction.data.tolist()[0][0])
                    current_epoch_records["actual"].append(label)
                    current_epoch_records["seqs"].append(len(df))

                    print ("Computing Loss & Backpropagation")
                    loss = criterion(prediction, label_t)
                    current_epoch_records["loss"].append(loss.data.tolist())
                    
                    print(f"File {i} / {len(dataset)}: Predicted Value: {prediction.data.tolist()[0][0]} ; True Value: {label}")
                    print(f"File {i} / {len(dataset)}: Loss: {loss.data.tolist()}")
                    print(f"Projected Completion Date: {projected_time(current_file_no, dataset.total_files(), e, total_epochs)}")

            pd.DataFrame(current_epoch_records).to_csv(outpath / "eval-records.csv", index = False)
            loss_record["test"].append(np.mean(current_epoch_records["loss"]))

            print (f"Epoch {e} / {total_epochs} - Completed.")
            print ("Epoch {} / {} - Average Training Loss:   {}".format(e, total_epochs, loss_record["train"]))
            print ("Epoch {} / {} - Average Validation Loss: {}".format(e, total_epochs, loss_record["val"]))
            print ("Saving Model Checkpoint.")
            torch.save(classifier.state_dict(), outpath / "classifier.pth")

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(f"Error Encountered: Logging Information", "ERROR")
        if torch.cuda.is_available():
            print(f"Torch Memory Taken: {torch.cuda.memory_allocated()}")
        print(f"Line {exc_tb.tb_lineno} - {type(e).__name__}: {str(e)}", "ERROR")
        
    except KeyboardInterrupt:
        print("Interrupted", "INFO")

    finally:
        try:
            make_directory_where_necessary(Path.cwd() / custom_configs["output-path"])
            torch.save(classifier.state_dict(), outpath / "classifier-trained.pth")
        except NameError:
            pass
