import glob
import zipfile
from pathlib import Path

dir = Path(__file__).resolve().parent

paths = ["embed/", "reduction/"]

files = []
for path in paths:
    files = files + list(glob.glob(path + "**", recursive=True))

with zipfile.ZipFile(dir / "upload.zip", "w") as zipf:
    for f in files:
        print(f"Zipping {f}")
        zipf.write(f)
