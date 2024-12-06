import json
import os
import subprocess
from pathlib import Path

encodings = [
    "atchley",
    "kidera",
    "rand",
    "aaprop",
    "tcrbert",
    "sceptr-default",
    "sceptr-tiny",
]
kfolds = 5

for i in range(kfolds):
    for encoding in encodings:
        fname = f"{encoding}-kfold-{i}.json"
        working_dir = "/cs/student/projects1/2020/cheuyuen/"
        command = f"python training/trainer.py -c {fname}\n"
        screen_start_command = f"screen -dmS {encoding}_{i}"
        subprocess.run(screen_start_command, shell=True, check=True)
        full_command = (
            f"cd {working_dir}; source tcr-embedding/bin/activate.csh; {command}"
        )
        screen_stuff_command = f'screen -S {encoding}_{i} -X stuff "{full_command}"'
        subprocess.run(screen_stuff_command, shell=True, check=True)
