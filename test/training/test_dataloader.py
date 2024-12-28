import os
import unittest

import numpy as np

from tcr_embeddings import runtime_constants
from tcr_embeddings.training import dataloader

os.chdir(runtime_constants.HOME_PATH)


class test_dataloader(unittest.TestCase):
    def test_ratio(self):
        ls_kfolds = list(range(5))
        for kf in ls_kfolds:
            loader = dataloader.Patients(
                split=runtime_constants.TRAIN_TEST_SPLIT,
                positives=runtime_constants.PATH_POSITIVE_CLASS,
                negatives=runtime_constants.PATH_NEGATIVE_CLASS,
                kfold=kf,
            )

            total_positives = len(
                list(
                    (
                        runtime_constants.HOME_PATH
                        / runtime_constants.PATH_POSITIVE_CLASS[0]
                    ).glob("*.tsv")
                )
            )
            total_negatives = len(
                list(
                    (
                        runtime_constants.HOME_PATH
                        / runtime_constants.PATH_NEGATIVE_CLASS[0]
                    ).glob("*.tsv")
                )
            )

            with open(
                runtime_constants.HOME_PATH
                / runtime_constants.PATH_POSITIVE_CLASS[0]
                / "kfold.txt"
            ) as f:
                total_positives_in_kfold = len(
                    f.readlines()[kf].replace("\n", "").split("<>")
                )

            with open(
                runtime_constants.HOME_PATH
                / runtime_constants.PATH_NEGATIVE_CLASS[0]
                / "kfold.txt"
            ) as f:
                total_negatives_in_kfold = len(
                    f.readlines()[kf].replace("\n", "").split("<>")
                )

            loader.train()
            pos_ratio = (total_positives - total_positives_in_kfold) / len(loader)
            neg_ratio = (total_negatives - total_negatives_in_kfold) / len(loader)

            self.assertEqual(pos_ratio, loader.ratio(1))
            self.assertEqual(neg_ratio, loader.ratio(0))

    def test_len(self):
        ls_positives = list(
            (
                runtime_constants.HOME_PATH / runtime_constants.PATH_POSITIVE_CLASS[0]
            ).glob("*.tsv")
        )
        ls_negatives = list(
            (
                runtime_constants.HOME_PATH / runtime_constants.PATH_NEGATIVE_CLASS[0]
            ).glob("*.tsv")
        )
        split_ratio = np.linspace(0, 1, 10)
        ls_kfolds = list(range(5))
        for split_val in split_ratio:
            for kf in ls_kfolds:
                loader = dataloader.Patients(
                    split=split_val,
                    positives=runtime_constants.PATH_POSITIVE_CLASS,
                    negatives=runtime_constants.PATH_NEGATIVE_CLASS,
                    kfold=kf,
                )
                self.assertEqual(
                    loader.total_files(), len(ls_positives) + len(ls_negatives)
                )

    def test_no_duplicates(self):
        split_ratio = np.linspace(0, 1, 10)
        ls_kfolds = list(range(5))
        for split_val in split_ratio:
            for kf in ls_kfolds:
                loader = dataloader.Patients(
                    split=split_val,
                    positives=runtime_constants.PATH_POSITIVE_CLASS,
                    negatives=runtime_constants.PATH_NEGATIVE_CLASS,
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
                    positives=runtime_constants.PATH_POSITIVE_CLASS,
                    negatives=runtime_constants.PATH_NEGATIVE_CLASS,
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
                    positives=runtime_constants.PATH_POSITIVE_CLASS,
                    negatives=runtime_constants.PATH_NEGATIVE_CLASS,
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
        ls_positives = list(
            (
                runtime_constants.HOME_PATH / runtime_constants.PATH_POSITIVE_CLASS[0]
            ).glob("*.tsv")
        )
        ls_negatives = list(
            (
                runtime_constants.HOME_PATH / runtime_constants.PATH_NEGATIVE_CLASS[0]
            ).glob("*.tsv")
        )
        split_ratio = np.linspace(0, 1, 10)
        ls_kfolds = list(range(5))
        for split_val in split_ratio:
            for kf in ls_kfolds:
                loader = dataloader.Patients(
                    split=split_val,
                    positives=runtime_constants.PATH_POSITIVE_CLASS,
                    negatives=runtime_constants.PATH_NEGATIVE_CLASS,
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


if __name__ == "__main__":
    unittest.main()
