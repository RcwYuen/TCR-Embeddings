import glob
import zipfile
from pathlib import Path

dirs = Path(__file__).resolve().parent

paths = ["embed/", "reduction/"]

files: list = []
for path in paths:
    files = files + list(glob.glob(path + "**", recursive=True))

with zipfile.ZipFile(dirs / "upload.zip", "w") as zipf:
    for f in files:
        print(f"Zipping {f}")
        zipf.write(f)
