from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm.notebook import tqdm

dct_filename_to_lens = {}
data_src = "results/"

for pth in Path.cwd().glob("data/*/cleaned/*.tsv"):
    dct_filename_to_lens[len(pd.read_csv(pth, sep="\t", dtype=str))] = pth

assert len(set(Counter(list(dct_filename_to_lens.values())).values())) == 1

ls_positives = []
ls_negatives = []

for method_kf in tqdm(list((Path.cwd() / data_src).glob("**/kfold-*"))):
    set_test_record_lens = set()
    for epoch in method_kf.glob("Epoch */test-records.csv"):
        set_test_record_lens.add(tuple(pd.read_csv(epoch)["seqs"]))
    assert len(set_test_record_lens) == 1

    ls_test_files = [
        str(dct_filename_to_lens[i].relative_to(Path.cwd()))
        for i in list(set_test_record_lens)[0]
    ]
    positives = "<>".join([i for i in ls_test_files if "LTX" in i])
    negatives = "<>".join([i for i in ls_test_files if "LTX" not in i])

    if positives not in ls_positives:
        ls_positives.append(positives)
    if negatives not in ls_negatives:
        ls_negatives.append(negatives)

    with open(method_kf / "pos0-kfold.txt", "w") as f:
        f.write(f"{positives}")
    with open(method_kf / "neg0-kfold.txt", "w") as f:
        f.write(f"{negatives}")

assert len(ls_positives) == 5
assert len(ls_negatives) == 5

with open(Path.cwd() / "data/Tx/cleaned/kfold.txt", "w") as f:
    f.write("\n".join(ls_positives))

with open(Path.cwd() / "data/tcvhcw/cleaned/kfold.txt", "w") as f:
    f.write("\n".join(ls_negatives))
