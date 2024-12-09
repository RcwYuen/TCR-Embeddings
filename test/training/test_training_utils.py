import os
import unittest
import warnings

import numpy as np
import torch

from tcr_embeddings import constants, runtime_constants
from tcr_embeddings.training import training_utils as utils


class test_train_utils(unittest.TestCase):
    LS_ENCODINGS = [
        "atchley",
        "kidera",
        "aaprop",
        "rand",
        "sceptr-tiny",
        "sceptr-default",
    ]
    LS_REDUCTION = ["autoencoder", "johnson-lindenstarauss", "no-reduction"]
    constants.CHANGE_NEGATIVE_CLASS_PATH(["test/data/false"], acknowledge=True)
    constants.CHANGE_POSITIVE_CLASS_PATH(["test/data/true"], acknowledge=True)
    constants.CHANGE_LR(1, acknowledge=True)

    def test_embedding_calculated_is_correct(self):
        for encoding_str in test_train_utils.LS_ENCODINGS:
            encoder = utils.get_embedding_method(encoding_str)
            for reduction_str in test_train_utils.LS_REDUCTION:
                temp_config = utils.load_configs(
                    {"encoding": encoding_str, "reduction": reduction_str}
                )

                reducer = utils.get_reduction(temp_config)

                ls_encoded = reducer.reduce(
                    encoder.calc_vector_representations(runtime_constants.DF_SAMPLE)
                ).tolist()[0]

                ls_encoded_test = utils.calculate_embeddings(
                    runtime_constants.DF_SAMPLE, temp_config
                ).tolist()[0]

                for i, j in zip(ls_encoded, ls_encoded_test):
                    self.assertTrue(abs(i - j) < 0.0001)

    def test_follows_constants_use_cuda_config(self):
        if torch.cuda.is_available():
            original_config = constants.USE_CUDA

            constants.CHANGE_USE_CUDA(True, acknowledge=True)
            tensor = utils.calculate_embeddings(
                runtime_constants.DF_SAMPLE, configs=utils.load_configs({})
            )
            self.assertNotEqual(tensor.get_device(), -1)
            # note: we cannot assert the device allocated will definitely be the same as on-the-fly allocating
            # a tensor, so we can only assert that it is not cpu

            constants.CHANGE_USE_CUDA(False, acknowledge=True)
            tensor = utils.calculate_embeddings(
                runtime_constants.DF_SAMPLE, configs=utils.load_configs({})
            )
            self.assertEqual(tensor.get_device(), -1)

            constants.CHANGE_USE_CUDA(original_config, acknowledge=True)

        else:
            warnings.warn("Cuda is not available, test cannot be ran.")

    def test_classifier_updates(self):
        logger = utils.create_logger(
            runtime_constants.HOME_PATH / "test/training", utils.parse_args()
        )
        utils.start_timer()
        configs = utils.load_configs({})
        classifier = utils.create_classifier(configs)
        optim = utils.create_optimiser(classifier)
        dataset = utils.create_dataset(configs)

        ls_old_classifier_params = [
            np.array(i.tolist()) for i in classifier.parameters()
        ]

        kwargs = {
            "dataset": dataset,
            "classifier": classifier,
            "configs": configs,
            "current_file_no": 0,
            "optim": optim,
            "current_epoch": 0,
        }

        dataset.train()
        classifier.train()
        _ = utils.iterate_through_files(**kwargs)  # type: ignore

        for old_params, new_params in zip(
            ls_old_classifier_params, classifier.parameters()
        ):
            self.assertEqual(old_params.shape, new_params.shape)
            self.assertNotEqual(old_params.tolist(), new_params.tolist())

        logger.close()
        os.remove(runtime_constants.HOME_PATH / "test/training/training.log")

    def test_iteration_rejects_unmatched_dataloader_model_mode(self):
        utils.create_logger(
            runtime_constants.HOME_PATH / "test/training", utils.parse_args()
        )
        utils.start_timer()
        configs = utils.load_configs({})
        classifier = utils.create_classifier(configs)
        optim = utils.create_optimiser(classifier)
        dataset = utils.create_dataset(configs)

        kwargs = {
            "dataset": dataset,
            "classifier": classifier,
            "configs": configs,
            "current_file_no": 0,
            "optim": optim,
            "current_epoch": 0,
        }

        dataset.test()
        classifier.train()
        with self.assertRaises(ValueError) as err1:
            _ = utils.iterate_through_files(**kwargs)  # type: ignore

        dataset.validation()
        classifier.train()
        with self.assertRaises(ValueError) as err2:
            _ = utils.iterate_through_files(**kwargs)  # type: ignore

        dataset.train()
        classifier.eval()
        with torch.no_grad():
            with self.assertRaises(ValueError) as err3:
                _ = utils.iterate_through_files(**kwargs)  # type: ignore

        for err in [err1, err2, err3]:
            self.assertEqual(
                str(err.exception), "Dataset mode and gradient mode does not match."
            )


if __name__ == "__main__":
    unittest.main()
