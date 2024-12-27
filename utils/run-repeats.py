import subprocess

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
        fname = f"{encoding}-kfold-{i}.json"  # change to your config names
        working_dir = "path/to/your/working/directory"
        command = f"python -m tcr_embeddings.trainer -c {fname}\n"
        screen_start_command = f"screen -dmS {encoding}_{i}"
        subprocess.run(screen_start_command, shell=True, check=True)
        full_command = f"cd {working_dir}; conda activate tcr_embeddings; {command}"
        screen_stuff_command = f'screen -S {encoding}_{i} -X stuff "{full_command}"'
        subprocess.run(screen_stuff_command, shell=True, check=True)
