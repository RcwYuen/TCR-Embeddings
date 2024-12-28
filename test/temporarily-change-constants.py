import json

dct_test_constants = {
    "PATH_POSITIVE_CLASS": ["test/data/true"],
    "PATH_NEGATIVE_CLASS": ["test/data/false"],
    "EPOCHS": 3,
    "LR": 1,
    "TRAIN_TEST_SPLIT": 1,
    "ACCUMMULATION": 2,
    "L2_PENALTY": 0,
    "RAND_SEED": 42,
    "USE_CUDA": False,
}

with open("tcr_embeddings/constants.json", "w") as f:
    f.writelines(json.dumps(dct_test_constants, indent=4))
