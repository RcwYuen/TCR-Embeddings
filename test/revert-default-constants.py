import json

dct_default = {
    "PATH_POSITIVE_CLASS": ["data/tcvhcw/cleaned"],
    "PATH_NEGATIVE_CLASS": ["data/Tx/cleaned"],
    "EPOCHS": 100,
    "LR": 0.025,
    "TRAIN_TEST_SPLIT": 1,
    "ACCUMMULATION": 4,
    "L2_PENALTY": 0.001,
    "RAND_SEED": 42,
    "USE_CUDA": True,
}

with open("tcr_embeddings/constants.json", "w") as f:
    f.writelines(json.dumps(dct_default, indent=4))
