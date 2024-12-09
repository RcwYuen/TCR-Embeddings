import unittest

from tcr_embeddings import runtime_constants
from tcr_embeddings.embed.embedder import Embedder
from tcr_embeddings.embed.llm import tcrbert


class test_llm(unittest.TestCase):
    LS_ALL_LLMS: list[Embedder] = [tcrbert()]
    LS_EXPT_OUTDIMS: list[int] = [768]

    def test_out_dim_is_as_expected(self):
        for embedder, outdim in zip(test_llm.LS_ALL_LLMS, test_llm.LS_EXPT_OUTDIMS):
            self.assertEqual(embedder.get_out_dim(), outdim)

    def test_vec_representation_in_correct_shape(self):
        for embedder, outdim in zip(test_llm.LS_ALL_LLMS, test_llm.LS_EXPT_OUTDIMS):
            representation = embedder.calc_vector_representations(
                runtime_constants.DF_SAMPLE, batchsize=1
            )
            self.assertEqual(representation.shape, (1, outdim))

    def test_vec_representation_in_correct_shape_harder(self):
        for embedder, outdim in zip(test_llm.LS_ALL_LLMS, test_llm.LS_EXPT_OUTDIMS):
            representation = embedder.calc_vector_representations(
                runtime_constants.DF_FULL, batchsize=128
            )
            self.assertEqual(
                representation.shape, (len(runtime_constants.DF_FULL), outdim)
            )


if __name__ == "__main__":
    unittest.main()
