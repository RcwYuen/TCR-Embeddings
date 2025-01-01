from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame


class Patients(torch.utils.data.Dataset):
    def __init__(
        self,
        split: float = 0.8,
        positives: list | None = None,
        negatives: list | None = None,
        kfold: int = 0,
    ) -> None:
        super(Patients, self).__init__()
        # 1 for training, 0 for validation, -1 for test
        if positives is None:
            positives = []
        if negatives is None:
            negatives = []

        self.__mode: int = 1
        self.__positives: List[str] = positives
        self.__negatives: List[str] = negatives
        self.__split: float = split
        self.__kfold: int = kfold
        self._ext: str = self._get_ext()

        self.__training_data: list = []
        self.__validation_data: list = []
        self.__testing_data: list = []
        # Ratio for `negatives: positives` in training data
        self.__ratio: list = [0, 0]

        self.__load_kfold(self.__positives, 1)
        self.__load_kfold(self.__negatives, 0)

        np.random.shuffle(self.__training_data)
        self.__data: list = [
            self.__validation_data,
            self.__training_data,
            self.__testing_data,
        ]
        self.__getfname = False

    def set_get_fname_mode(self, state: bool) -> None:
        self.__getfname = state

    def total_files(self) -> int:
        return sum(len(i) for i in self.__data)

    def files_by_category(self) -> dict:
        return {
            "Training": len(self.__training_data),
            "Validation": len(self.__validation_data),
            "Testing": len(self.__testing_data),
        }

    def all_files_by_category(self) -> dict:
        return {
            "Training": self.__training_data,
            "Validation": self.__validation_data,
            "Testing": self.__testing_data,
        }

    def get_ext(self) -> str:
        return self._ext

    def __len__(self) -> int:
        return len(self.__data[self.__mode])

    def _open_file(self, fname: str) -> pd.DataFrame:
        match self._ext:
            case ".tsv":
                return pd.read_csv(fname, sep="\t", dtype=str)

            case ".pq":
                return pd.read_parquet(fname)

            case _:
                raise ValueError("File Extension not supported.")

    def _get_ext(self):
        set_exts: set = set()
        for pth in self.__positives + self.__negatives:
            set_exts = set_exts.union(
                set(i.suffix for i in (Path.cwd() / pth).glob("*"))
            )
        ls_exts: list = [i for i in set_exts if i != ".txt"]
        assert len(ls_exts) == 1, "Expected only 1 file type within directory"
        return ls_exts[0]

    def __getitem__(self, idx: int) -> list[DataFrame | Any]:
        label, fname = self.__data[self.__mode][idx]
        if self.__getfname:
            return [label, fname, self._open_file(fname)]
        else:
            return [label, self._open_file(fname)]

    def __load_kfold(self, directories: list, label: int) -> None:
        for directory in directories:
            set_files = set((Path.cwd() / directory).glob("*" + self._ext))

            with open(Path.cwd() / directory / "kfold.txt", "r") as f:
                set_kfold = {
                    Path.cwd() / directory / i
                    for i in f.readlines()[self.__kfold].replace("\n", "").split("<>")
                }

            ls_tsvs = list(set_files - set_kfold)

            ls_train_idx = np.random.choice(
                np.arange(len(ls_tsvs)),
                replace=False,
                size=int(len(ls_tsvs) * self.__split),
            ).astype(int)
            ls_val_idx = np.setdiff1d(np.arange(len(ls_tsvs)), ls_train_idx).astype(int)

            self.__training_data += [(label, ls_tsvs[i]) for i in ls_train_idx]
            self.__validation_data += [(label, ls_tsvs[i]) for i in ls_val_idx]
            self.__testing_data += [(label, file) for file in set_kfold]
            self.__ratio[label] += len(ls_train_idx)

    def ratio(self, label: int) -> float:
        return self.__ratio[label] / sum(self.__ratio)

    def train(self) -> None:
        self.__mode = 1

    def test(self) -> None:
        self.__mode = -1

    def validation(self) -> None:
        self.__mode = 0

    def get_mode(self) -> str:
        match self.__mode:
            case 1:
                return "train"
            case -1:
                return "test"
            case 0:
                return "eval"

            case _:
                raise Exception("Internal Variable '__mode' is corrupted.")
