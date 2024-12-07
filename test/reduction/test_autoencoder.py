import unittest

import numpy as np
import torch
from sceptr import variant

from tcr_embeddings.embed.llm import tcrbert
from tcr_embeddings.embed.physicochemical import aaprop, atchley, kidera, rand
from tcr_embeddings.reduction.reduction import AutoEncoder


class PassThrough:
    pass


class test_ae(unittest.TestCase):
    LS_METHODS = [
        (atchley(), "atchley"),
        (aaprop(), "aaprop"),
        (rand(), "rand"),
        (kidera(), "kidera"),
        (tcrbert(), "tcrbert"),
        (variant.default(), "sceptr-default"),
        (variant.tiny(), "sceptr-tiny"),
    ]

    def test_accepts_valid_encoding_method(self):
        for method in test_ae.LS_METHODS:
            try:
                _ = AutoEncoder(*method)
            except AssertionError:
                self.fail("This should not raise an Assertion Error.")

    def test_rejects_invalid_encoding_method(self):
        with self.assertRaises(AssertionError) as err:
            AutoEncoder(PassThrough, "PassThrough")

        self.assertEqual(
            str(err.exception),
            "Encoding Method passed is not an eligible encoding method.",
        )

    def test_norm_of_reduced_ndarray_is_one(self):
        for method in test_ae.LS_METHODS:
            reduction_method = AutoEncoder(*method)
            in_dim = reduction_method.get_in_dim()
            ls_reduced = reduction_method.reduce(
                np.random.normal(0, 1, size=(100, in_dim))
            )
            ls_diffs = abs(np.linalg.norm(ls_reduced, axis=1) - 1).tolist()
            for val in ls_diffs:
                self.assertTrue(val < 0.001)

    def test_norm_of_reduced_tensor_is_one(self):
        for method in test_ae.LS_METHODS:
            reduction_method = AutoEncoder(*method)
            in_dim = reduction_method.get_in_dim()
            ls_vecs = np.random.normal(0, 1, size=(100, in_dim))
            ls_reduced = reduction_method.reduce(torch.from_numpy(ls_vecs))
            ls_diffs = abs(np.linalg.norm(ls_reduced, axis=1) - 1).tolist()
            for val in ls_diffs:
                self.assertTrue(val < 0.001)

    def test_ndarray_norming_right_axis(self):
        for method in test_ae.LS_METHODS:
            reduction_method = AutoEncoder(*method)
            in_dim = reduction_method.get_in_dim()
            ls_reduced = reduction_method.reduce(
                np.random.normal(0, 1, size=(100, in_dim))
            )
            ls_diffs = abs(np.linalg.norm(ls_reduced, axis=1) - 1).tolist()
            self.assertTrue(len(ls_diffs), 100)

    def test_tensor_norming_right_axis(self):
        for method in test_ae.LS_METHODS:
            reduction_method = AutoEncoder(*method)
            in_dim = reduction_method.get_in_dim()
            ls_vecs = np.random.normal(0, 1, size=(100, in_dim))
            ls_reduced = reduction_method.reduce(torch.from_numpy(ls_vecs))
            ls_diffs = abs(np.linalg.norm(ls_reduced, axis=1) - 1).tolist()
            self.assertTrue(len(ls_diffs), 100)


if __name__ == "__main__":
    unittest.main()
