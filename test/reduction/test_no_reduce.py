import unittest

import numpy as np
import torch

from tcr_embeddings.reduction.reduction import NoReduce


class test_no_reduce(unittest.TestCase):
    LS_DIMS: list = [10, 14, 16, 64, 768]

    def test_norm_of_reduced_ndarray_is_one(self):
        for in_dim in test_no_reduce.LS_DIMS:
            reduction_method = NoReduce()
            ls_reduced = reduction_method.reduce(
                np.random.normal(0, 1, size=(100, in_dim))
            )
            ls_diffs = abs(np.linalg.norm(ls_reduced, axis=1) - 1).tolist()
            for val in ls_diffs:
                self.assertTrue(val < 0.001)

    def test_norm_of_reduced_tensor_is_one(self):
        for in_dim in test_no_reduce.LS_DIMS:
            reduction_method = NoReduce()
            ls_vecs = np.random.normal(0, 1, size=(100, in_dim))
            ls_reduced = reduction_method.reduce(torch.from_numpy(ls_vecs))
            ls_diffs = abs(np.linalg.norm(ls_reduced, axis=1) - 1).tolist()
            for val in ls_diffs:
                self.assertTrue(val < 0.001)

    def test_ndarray_norming_right_axis(self):
        for in_dim in test_no_reduce.LS_DIMS:
            reduction_method = NoReduce()
            ls_reduced = reduction_method.reduce(
                np.random.normal(0, 1, size=(100, in_dim))
            )
            ls_diffs = abs(np.linalg.norm(ls_reduced, axis=1) - 1).tolist()
            self.assertTrue(len(ls_diffs), 100)

    def test_tensor_norming_right_axis(self):
        for in_dim in test_no_reduce.LS_DIMS:
            reduction_method = NoReduce()
            ls_vecs = np.random.normal(0, 1, size=(100, in_dim))
            ls_reduced = reduction_method.reduce(torch.from_numpy(ls_vecs))
            ls_diffs = abs(np.linalg.norm(ls_reduced, axis=1) - 1).tolist()
            self.assertTrue(len(ls_diffs), 100)


if __name__ == "__main__":
    unittest.main()
