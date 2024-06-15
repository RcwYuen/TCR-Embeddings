import torch
import pandas as pd
import random
from pathlib import Path
import numpy as np

class Patients(torch.utils.data.Dataset):
    def __init__(
        self,
        split = 0.8,
        positives = [],
        negatives = [],
        kfold = 0
    ):
        super(Patients, self).__init__()
        self.__mode = 1 # 1 for training, 0 for validation, -1 for test
        self.__positives = positives
        self.__negatives = negatives
        self.__split = split
        self.__kfold = kfold

        self.__training_data = []
        self.__ratio = [0, 0] # Ratio for `negatives: positives` in training data
        self.__validation_data = []
        self.__testing_data = []

        self.__load_kfold(self.__positives, 1)
        self.__load_kfold(self.__negatives, 0)

        np.random.shuffle(self.__training_data)
    
    def total_files(self):
        return len(self.__training_data) + len(self.__validation_data) + len(self.__testing_data)

    def files_by_category(self):
        return {
            "Training": len(self.__training_data), 
            "Validation": len(self.__validation_data), 
            "Testing": len(self.__testing_data)
        }

    def __len__(self):
        if self.__mode == 1: # Training
            return len(self.__training_data)
        elif self.__mode == 0: # Validation
            return len(self.__validation_data)
        elif self.__mode == -1: # Test
            return len(self.__testing_data)        

    def __getitem__(self, idx):
        if self.__mode == 1: # Training
            file = self.__training_data[idx]
        elif self.__mode == 0: # Validation
            file = self.__validation_data[idx]
        elif self.__mode == -1: # Test
            file = self.__testing_data[idx]
        
        return [file[0], pd.read_csv(file[1], sep = "\t", dtype = str)]

    def __load_kfold(self, dirs, label):
        for dir in dirs:
            tsvs = set((Path.cwd() / dir).glob("*.tsv"))
            kfold = [
                i.replace("\n", "").split("<>") for i in open(Path.cwd() / dir / "kfold.txt", "r").readlines()
            ][self.__kfold]
            tsvs = list(tsvs - set([Path.cwd() / i for i in kfold]))

            trainidx = np.random.choice(np.arange(len(tsvs)), replace = False, size = int(len(tsvs) * self.__split)).astype(int)
            validationidx = np.setdiff1d(np.arange(len(tsvs)), trainidx).astype(int)

            self.__training_data += [(label, tsvs[i]) for i in trainidx]
            self.__validation_data += [(label, tsvs[i]) for i in validationidx]
            self.__ratio[label] += len(trainidx)
            self.__testing_data += [(label, Path.cwd() / file) for file in kfold]
    
    def ratio(self, label):
        return self.__ratio[label] / sum(self.__ratio)

    def train(self):
        self.__mode = 1
    
    def test(self):
        self.__mode = -1

    def validation(self):
        self.__mode = 0