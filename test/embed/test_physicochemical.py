import unittest
from tcr_embeddings.embed.physicochemical import atchley, kidera, aaprop, rand
from tcr_embeddings.embed.embedder import Embedder
from tcr_embeddings import runtime_constants
import pandas as pd
import numpy as np

class test_physicochemical(unittest.TestCase):
    LS_ALL_PHYSICOEMBEDDERS: list[Embedder] = [
        atchley(), kidera(), aaprop(), rand()
    ]
    LS_EXPT_OUTDIMS: list[int] = [
        5, 10, 14, 5
    ]

    # THE FOLLOWING ARE EXPECTED OUTPUTS OF EMBEDDING AMINO ACID 'CAT'
    prot = "CAT"
    LS_EXPT_EMBEDDED: list[np.ndarray] = [
        atchley().embedding_space.loc[list(prot)].mean(axis=0).values,
        kidera().embedding_space.loc[list(prot)].mean(axis=0).values,
        aaprop().embedding_space.loc[list(prot)].mean(axis=0).values,
        rand().embedding_space.loc[list(prot)].mean(axis=0).values
    ]

    DF_SAMPLE_INPUT = pd.DataFrame([None, None, None, prot], index=["TRAV", "CDR3A", "TRBV", "CDR3B"]).T


    def test_out_dim_is_as_expected(self):
        for embedder, outdim in zip(test_physicochemical.LS_ALL_PHYSICOEMBEDDERS, test_physicochemical.LS_EXPT_OUTDIMS):
            self.assertEqual(embedder.get_out_dim(), outdim)

    def test_vec_representation_in_correct_shape(self):
        df_sample = pd.read_csv(runtime_constants.DATA_PATH / "sample.tsv", sep = "\t")
        for embedder, outdim in zip(test_physicochemical.LS_ALL_PHYSICOEMBEDDERS, test_physicochemical.LS_EXPT_OUTDIMS):
            representation = embedder.calc_vector_representations(df_sample, batchsize = 1)
            self.assertEqual(representation.shape, (1, outdim))

    def test_vec_representation_in_correct_shape_harder(self):
        df_full = pd.read_csv(runtime_constants.DATA_PATH / "full.tsv", sep = "\t")
        for embedder, outdim in zip(test_physicochemical.LS_ALL_PHYSICOEMBEDDERS, test_physicochemical.LS_EXPT_OUTDIMS):
            representation = embedder.calc_vector_representations(df_full, batchsize=1)
            self.assertEqual(representation.shape, (len(df_full), outdim))

    def test_embedded_is_as_expected(self):
        for embedder, expt_outcome in zip(
                test_physicochemical.LS_ALL_PHYSICOEMBEDDERS,
                test_physicochemical.LS_EXPT_EMBEDDED):
            ls_embedded = embedder.calc_vector_representations(test_physicochemical.DF_SAMPLE_INPUT).tolist()[0]

            for e1, e2 in zip(ls_embedded, expt_outcome.tolist()):
                self.assertTrue(abs(e1 - e2) < 0.0001)


if __name__ == '__main__':
    unittest.main()