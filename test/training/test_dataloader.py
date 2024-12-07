import os
import unittest
from pathlib import Path

import numpy as np
from tcr_embeddings.training import dataloader
from tcr_embeddings import runtime_constants

os.chdir(runtime_constants.HOME_PATH)


class test_dataloader(unittest.TestCase):
    def test_ratio(self):
        ls_kfolds = list(range(5))
        for kf in ls_kfolds:
            loader = dataloader.Patients(
                split=1,
                positives=["data/Tx/cleaned"],
                negatives=["data/tcvhcw/cleaned"],
                kfold=kf,
            )

            total_positives = len(list((Path.cwd() / "data/Tx/cleaned").glob("*.tsv")))
            total_negatives = len(
                list((Path.cwd() / "data/tcvhcw/cleaned").glob("*.tsv"))
            )

            with open(Path.cwd() / "data/Tx/cleaned/kfold.txt") as f:
                total_positives_in_kfold = len(
                    f.readlines()[kf].replace("\n", "").split("<>")
                )

            with open(Path.cwd() / "data/tcvhcw/cleaned/kfold.txt") as f:
                total_negatives_in_kfold = len(
                    f.readlines()[kf].replace("\n", "").split("<>")
                )

            loader.train()
            pos_ratio = (total_positives - total_positives_in_kfold) / len(loader)
            neg_ratio = (total_negatives - total_negatives_in_kfold) / len(loader)

            self.assertEqual(pos_ratio, loader.ratio(1))
            self.assertEqual(neg_ratio, loader.ratio(0))

    def test_len(self):
        total_positives = len(list((Path.cwd() / "data/Tx/cleaned").glob("*.tsv")))
        total_negatives = len(list((Path.cwd() / "data/tcvhcw/cleaned").glob("*.tsv")))
        split_ratio = np.linspace(0, 1, 10)
        ls_kfolds = list(range(5))
        for split_val in split_ratio:
            for kf in ls_kfolds:
                loader = dataloader.Patients(
                    split=split_val,
                    positives=["data/Tx/cleaned"],
                    negatives=["data/tcvhcw/cleaned"],
                    kfold=kf,
                )
                self.assertEqual(
                    loader.total_files(), total_positives + total_negatives
                )

    def test_no_duplicates(self):
        split_ratio = np.linspace(0, 1, 10)
        ls_kfolds = list(range(5))
        for split_val in split_ratio:
            for kf in ls_kfolds:
                loader = dataloader.Patients(
                    split=split_val,
                    positives=["data/Tx/cleaned"],
                    negatives=["data/tcvhcw/cleaned"],
                    kfold=kf,
                )
                train_set = loader.all_files_by_category()["Training"]
                val_set = loader.all_files_by_category()["Validation"]
                test_set = loader.all_files_by_category()["Testing"]

                self.assertEqual(sorted(train_set), sorted(list(set(train_set))))
                self.assertEqual(sorted(val_set), sorted(list(set(val_set))))
                self.assertEqual(sorted(test_set), sorted(list(set(test_set))))

    def test_no_leakage(self):
        split_ratio = np.linspace(0, 1, 10)
        ls_kfolds = list(range(5))
        for split_val in split_ratio:
            for kf in ls_kfolds:
                loader = dataloader.Patients(
                    split=split_val,
                    positives=["data/Tx/cleaned"],
                    negatives=["data/tcvhcw/cleaned"],
                    kfold=kf,
                )
                train_set = set(loader.all_files_by_category()["Training"])
                val_set = set(loader.all_files_by_category()["Validation"])
                test_set = set(loader.all_files_by_category()["Testing"])

                self.assertEqual(train_set.intersection(val_set), set())
                self.assertEqual(train_set.intersection(test_set), set())
                self.assertEqual(val_set.intersection(test_set), set())

    def test_mode_correct(self):
        split_ratio = np.linspace(0, 1, 10)
        ls_kfolds = list(range(5))
        for split_val in split_ratio:
            for kf in ls_kfolds:
                loader = dataloader.Patients(
                    split=split_val,
                    positives=["data/Tx/cleaned"],
                    negatives=["data/tcvhcw/cleaned"],
                    kfold=kf,
                )
                train_set = set(loader.all_files_by_category()["Training"])
                val_set = set(loader.all_files_by_category()["Validation"])
                test_set = set(loader.all_files_by_category()["Testing"])

                loader.train()
                self.assertEqual(len(train_set), len(loader))

                loader.validation()
                self.assertEqual(len(val_set), len(loader))

                loader.test()
                self.assertEqual(len(test_set), len(loader))

    def test_correct_label_assigned(self):
        ls_positives = list((Path.cwd() / "data/Tx/cleaned").glob("*.tsv"))
        ls_negatives = list((Path.cwd() / "data/tcvhcw/cleaned").glob("*.tsv"))
        split_ratio = np.linspace(0, 1, 10)
        ls_kfolds = list(range(5))
        for split_val in split_ratio:
            for kf in ls_kfolds:
                loader = dataloader.Patients(
                    split=split_val,
                    positives=["data/Tx/cleaned"],
                    negatives=["data/tcvhcw/cleaned"],
                    kfold=kf,
                )
                data = sorted(
                    loader.all_files_by_category()["Training"]
                    + loader.all_files_by_category()["Validation"]
                    + loader.all_files_by_category()["Testing"]
                )

                for i in data:
                    self.assertTrue(i[0] in [0, 1])

                    if i[0] == 0:
                        self.assertTrue(i[1] in ls_negatives)

                    elif i[0] == 1:
                        self.assertTrue(i[1] in ls_positives)

    def test_kfolds(self):
        with open(Path.cwd() / "data/Tx/cleaned/kfold.txt") as f:
            ls_pos_kfolds = "".join(f.readlines()).replace("\n", "<>").split("<>")

        with open(Path.cwd() / "data/tcvhcw/cleaned/kfold.txt") as f:
            ls_neg_kfolds = "".join(f.readlines()).replace("\n", "<>").split("<>")

        self.assertEqual(
            sorted([Path.cwd() / i for i in ls_pos_kfolds]),
            sorted(list((Path.cwd() / "data/Tx/cleaned").glob("*.tsv"))),
        )

        self.assertEqual(
            sorted([Path.cwd() / i for i in ls_neg_kfolds]),
            sorted(list((Path.cwd() / "data/tcvhcw/cleaned").glob("*.tsv"))),
        )


if __name__ == "__main__":
    unittest.main()
