from pathlib import Path
import re
import time

def get_epoch(base_path):
    max_epoch = -1
    pattern = re.compile(r'^Epoch (\d+)$')

    for folder in base_path.iterdir():
        if folder.is_dir():
            match = pattern.match(folder.name)
            if match:
                epoch_number = int(match.group(1))
                max_epoch = max(max_epoch, epoch_number)
    return max_epoch

def stale_flag(cur_epoch, p):
    curr_epoch = (p / f"Epoch {cur_epoch}").stat().st_mtime
    return (time.time() - curr_epoch) / 60


dir = Path.cwd() / "results"
outstrs = []
for p in dir.glob("*/kfold-*"):
    curepoch = get_epoch(p)
    done = str((p / "classifier-trained.pth").exists())
    stale = round(stale_flag(curepoch, p))
    if not (p / f"Epoch {curepoch}" / "train-records.csv").exists():
        status = "Training"
    elif not (p / f"Epoch {curepoch}" / "eval-records.csv").exists():
        status = "Validating"
    else:
        status = "Testing"

    outstrs.append(f"{str(p.relative_to(dir)):40}  |  Current Epoch {curepoch:3}  |  Done: {done:5}  |  Status: {status:8}  |  Stale: {stale} mins")
   
outstrs = "\n".join(sorted(outstrs))
print (outstrs)