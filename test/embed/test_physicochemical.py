import unittest

import pandas as pd

from tcr_embeddings import runtime_constants
from tcr_embeddings.embed.embedder import Embedder
from tcr_embeddings.embed.physicochemical import aaprop, atchley, kidera, rand


class test_physicochemical(unittest.TestCase):
    LS_ALL_PHYSICOEMBEDDERS: list[Embedder] = [atchley(), kidera(), aaprop(), rand()]
    LS_EXPT_OUTDIMS: list[int] = [5, 10, 14, 5]

    # THE FOLLOWING ARE EXPECTED OUTPUTS OF EMBEDDING AMINO ACID 'CAT'
    prot = "CAT"
    LS_EXPT_EMBEDDED: list[list[float]] = [
        atchley().embedding_space.loc[list(prot)].mean(axis=0).tolist(),
        kidera().embedding_space.loc[list(prot)].mean(axis=0).tolist(),
        aaprop().embedding_space.loc[list(prot)].mean(axis=0).tolist(),
        rand().embedding_space.loc[list(prot)].mean(axis=0).tolist(),
    ]

    DF_SAMPLE_INPUT = pd.DataFrame(
        [None, None, None, prot], index=["TRAV", "CDR3A", "TRBV", "CDR3B"]
    ).T

    def test_out_dim_is_as_expected(self):
        for embedder, outdim in zip(
            test_physicochemical.LS_ALL_PHYSICOEMBEDDERS,
            test_physicochemical.LS_EXPT_OUTDIMS,
        ):
            self.assertEqual(embedder.get_out_dim(), outdim)

    def test_vec_representation_in_correct_shape(self):
        for embedder, outdim in zip(
            test_physicochemical.LS_ALL_PHYSICOEMBEDDERS,
            test_physicochemical.LS_EXPT_OUTDIMS,
        ):
            representation = embedder.calc_vector_representations(
                runtime_constants.DF_SAMPLE, batchsize=1
            )
            self.assertEqual(representation.shape, (1, outdim))

    def test_vec_representation_in_correct_shape_harder(self):
        for embedder, outdim in zip(
            test_physicochemical.LS_ALL_PHYSICOEMBEDDERS,
            test_physicochemical.LS_EXPT_OUTDIMS,
        ):
            representation = embedder.calc_vector_representations(
                runtime_constants.DF_FULL, batchsize=1
            )
            self.assertEqual(
                representation.shape, (len(runtime_constants.DF_FULL), outdim)
            )

    def test_embedded_is_as_expected(self):
        for embedder, expt_outcome in zip(
            test_physicochemical.LS_ALL_PHYSICOEMBEDDERS,
            test_physicochemical.LS_EXPT_EMBEDDED,
        ):
            ls_embedded: list[float] = embedder.calc_vector_representations(
                test_physicochemical.DF_SAMPLE_INPUT
            ).tolist()[0]

            for e1, e2 in zip(ls_embedded, expt_outcome):
                self.assertTrue(abs(e1 - e2) < 0.0001)


if __name__ == "__main__":
    unittest.main()
