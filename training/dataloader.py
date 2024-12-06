import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


class Patients(torch.utils.data.Dataset):
    def __init__(
        self,
        split: float = 0.8,
        positives: list = [],
        negatives: list = [],
        kfold: int = 0,
    ) -> None:
        super(Patients, self).__init__()
        # 1 for training, 0 for validation, -1 for test
        self.__mode: int = 1
        self.__positives: list = positives
        self.__negatives: list = negatives
        self.__split: float = split
        self.__kfold: int = kfold

        self.__training_data: list = []
        # Ratio for `negatives: positives` in training data
        self.__ratio: list = [0, 0]
        self.__validation_data: list = []
        self.__testing_data: list = []

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

    def __len__(self) -> int:
        return len(self.__data[self.__mode])

    def __getitem__(self, idx) -> int:
        label, fname = self.__data[self.__mode][idx]
        return [label, pd.read_csv(fname, sep="\t", dtype=str)]

    def __load_kfold(self, dirs: list, label: int) -> None:
        for dir in dirs:
            tsvs = set((Path.cwd() / dir).glob("*.tsv"))
            kfold = [
                i.replace("\n", "").split("<>")
                for i in open(Path.cwd() / dir / "kfold.txt", "r").readlines()
            ][self.__kfold]
            tsvs = list(tsvs - set([Path.cwd() / i for i in kfold]))

            trainidx = np.random.choice(
                np.arange(len(tsvs)), replace=False, size=int(len(tsvs) * self.__split)
            ).astype(int)
            validationidx = np.setdiff1d(np.arange(len(tsvs)), trainidx).astype(int)

            self.__training_data += [(label, tsvs[i]) for i in trainidx]
            self.__validation_data += [(label, tsvs[i]) for i in validationidx]
            self.__ratio[label] += len(trainidx)
            self.__testing_data += [(label, Path.cwd() / file) for file in kfold]

    def ratio(self, label: int) -> float:
        return self.__ratio[label] / sum(self.__ratio)

    def train(self) -> None:
        self.__mode = 1

    def test(self) -> None:
        self.__mode = -1

    def validation(self) -> None:
        self.__mode = 0
