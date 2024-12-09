# Training Configs
PATH_POSITIVE_CLASS: list[str] = ["data/Tx/cleaned"]
PATH_NEGATIVE_CLASS: list[str] = ["data/tcvhcw/cleaned"]
EPOCHS: int = 50
LR: float = 1e-3
TRAIN_TEST_SPLIT: int = 1  # keep this as 1 as we are running k-fold
ACCUMMULATION: int = 4
L2_PENALTY: float = 1e-4
RAND_SEED: int = 42
USE_CUDA: bool = True

# DANGEROUS ZONE - USED ONLY FOR UNITTEST
def CHANGE_USE_CUDA(change_to: bool, acknowledge: bool = False):
    if acknowledge:
        global USE_CUDA
        USE_CUDA = change_to
    else:
        raise NameError("You should not change constants.")

def CHANGE_POSITIVE_CLASS_PATH(change_to = list[str], acknowledge: bool = False):
    if acknowledge:
        global PATH_POSITIVE_CLASS
        PATH_POSITIVE_CLASS = change_to
    else:
        raise NameError("You should not change constants.")

def CHANGE_NEGATIVE_CLASS_PATH(change_to = list[str], acknowledge: bool = False):
    if acknowledge:
        global PATH_NEGATIVE_CLASS
        PATH_NEGATIVE_CLASS = change_to
    else:
        raise NameError("You should not change constants.")

def CHANGE_LR(change_to: float, acknowledge: bool = False):
    if acknowledge:
        global LR
        LR = change_to
    else:
        raise NameError("You should not change constants.")

"""
configs = {
        "positive-path": ["data"],
        "negative-path": ["data"],
        "output-path": "results",
        "epoch": 100,
        "lr": 1e-3,
        "kfold": 0,
        "train-split": 0.8,
        "accummulation": 4,
        "l2-penalty": 0,
        "encoding": "atchley",
        "reduction": "johnson-lindenstarauss",
        "seed": 42,
        "use-cuda": True,
    }
"""
