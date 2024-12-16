import unittest

from tcr_embeddings.interpret import interpretability


class test_find_best_auc(unittest.TestCase):
    def test_linearly_increasing_auc(self):
        ls_aucs = [(i, (0.1 * (i + 1))) for i in range(10)]
        best_epoch, _ = interpretability._find_best_epoch(ls_aucs)
        self.assertEqual(best_epoch, 9)

    def test_linearly_decreasing_auc(self):
        ls_aucs = [(i, (1 - 0.1 * (i + 1))) for i in range(10)]
        best_epoch, _ = interpretability._find_best_epoch(ls_aucs)
        self.assertEqual(best_epoch, 0)

    def test_same_start_and_end(self):
        ls_aucs = [
            (0, 1),
            (1, 0.9),
            (2, 0.8),
            (3, 0.7),
            (4, 0.6),
            (5, 0.5),
            (6, 0.6),
            (7, 0.7),
            (8, 0.8),
            (9, 1),
        ]
        best_epoch, _ = interpretability._find_best_epoch(ls_aucs)
        self.assertEqual(best_epoch, 9)


if __name__ == "__main__":
    unittest.main()
