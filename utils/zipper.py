import glob
import zipfile
from pathlib import Path

ls_paths = [
    "tcr_embeddings/",
    "data/tcvhcw/cleaned/",
    "data/Tx/cleaned/",
    "data/tcvhcw/embedded/",
    "data/Tx/embedded/",
    "utils/",
    "data/sample.tsv",
    "data/full.tsv",
]
ls_files = ["poetry.lock", "pyproject.toml"]

files: list = sum(
    [list(glob.glob(path + "**", recursive=True)) for path in ls_paths], ls_files
)

with zipfile.ZipFile(Path.cwd() / "upload.zip", "w") as zipf:
    for f in files:
        print(f"Zipping {f}")
        zipf.write(f)
