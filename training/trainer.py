from pathlib import Path
from dotenv import load_dotenv
import os, sys

dir = Path(__file__).resolve().parent
load_dotenv(Path.cwd() / ".env")
python_path = os.getenv('PYTHONPATH')
if python_path:
    sys.path.append(python_path)

import argparse
import json
import torch
from training.logger import Logger

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
        "kfold": 1,
        "train-split": 0.8,
        "accummulation": 4,
        "l2-penalty": 0,
        "encoding": "atchley"
    }

    if write:
        with open("config.json", "w") as outs:
            outs.write(json.dumps(configs))

    return configs

def load_configs(custom_configs):
    configs = defaults(write = False)
    for key, val in custom_configs.items():
        print(f"Config: {key}: {val}", "INFO")
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
    return True


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
                
        config_file = (
            parser.config_file if parser.config_file is not None else "config.json"
        )

        if torch.cuda.is_available():
            print(f"Torch CUDA Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")

        # In case config.json does not exist
        custom_configs = defaults(write_to=False) 
        custom_configs = load_configs(json.load(open(config_file, "r")))

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(f"Error Encountered: Logging Information", "ERROR")
        if torch.cuda.is_available():
            print(f"Torch Memory Taken: {torch.cuda.memory_allocated()}")
        print(f"Line {exc_tb.tb_lineno} - {type(e).__name__}: {str(e)}", "ERROR")
    
        
    except KeyboardInterrupt:
        print("Interrupted", "INFO")

'''
    finally:
        try:
            outpath = make_output_path(custom_configs["output-path"])
            make_directory_where_necessary(outpath)
            log.close()
            pd.DataFrame(trainloss).to_csv(outpath / "trainloss.csv", index = False, header = False)
            pd.DataFrame(trainacc).to_csv(outpath / "trainacc.csv", index = False, header = False)
            torch.save(bertmodel.state_dict(), outpath / "bertmodel-trained.pth")
            torch.save(classifier_model.state_dict(), outpath / "classifier-trained.pth")
        except NameError:
            pass
'''
