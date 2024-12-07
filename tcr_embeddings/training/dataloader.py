from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from virtualenv.config.convert import NoneType


class Patients(torch.utils.data.Dataset):
    def __init__(
        self,
        split: float = 0.8,
        positives: list | NoneType = None,
        negatives: list | NoneType = None,
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

    def __len__(self) -> int:
        return len(self.__data[self.__mode])

    def __getitem__(self, idx: int) -> list[DataFrame | Any]:
        label, fname = self.__data[self.__mode][idx]
        return [label, pd.read_csv(fname, sep="\t", dtype=str)]

    def __load_kfold(self, directories: list, label: int) -> None:
        for directory in directories:
            set_tsvs = set((Path.cwd() / directory).glob("*.tsv"))

            with open(Path.cwd() / directory / "kfold.txt", "r") as f:
                set_kfold = {
                    Path.cwd() / i
                    for i in f.readlines()[self.__kfold].replace("\n", "").split("<>")
                }

            set_tsvs = list(set_tsvs - set_kfold)

            ls_train_idx = np.random.choice(
                np.arange(len(set_tsvs)),
                replace=False,
                size=int(len(set_tsvs) * self.__split),
            ).astype(int)
            ls_val_idx = np.setdiff1d(np.arange(len(set_tsvs)), ls_train_idx).astype(
                int
            )

            self.__training_data += [(label, set_tsvs[i]) for i in ls_train_idx]
            self.__validation_data += [(label, set_tsvs[i]) for i in ls_val_idx]
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
