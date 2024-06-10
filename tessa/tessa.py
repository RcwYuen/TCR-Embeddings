import numpy as np
import pandas as pd
from keras.models import load_model
from keras.models import Model
from pathlib import Path

model_dir = Path.cwd() / "TrainedEncoder.h5"
aa_dict_dir = Path.cwd() / "Atchley_factors.csv"
encode_dim=80

def aamapping(peptideSeq,aa_dict,encode_dim):
    peptideArray = np.zeros((80, 5))
    if len(peptideSeq)>encode_dim:
        print('Length: '+str(len(peptideSeq))+' over bound!')
        peptideSeq=peptideSeq[0:encode_dim]        
    for idx, aa_single in enumerate(peptideSeq):
        try:
            peptideArray[idx] = aa_dict.loc[aa_single].to_numpy()
        except KeyError:
            pass
    return peptideArray

def datasetMap(dataset,aa_dict,encode_dim):
    return np.array([aamapping(seq, aa_dict, encode_dim) for seq in dataset])

def embed(df):
    tcr = [i for i in df["CDR3A"].tolist() + df["CDR3B"].tolist() if not pd.isna(i)]
    aa_dict = pd.read_csv(aa_dict_dir).set_index("Amino acid")
    TCR_contigs=datasetMap(tcr,aa_dict,encode_dim)
    TCR_contigs=TCR_contigs.reshape(-1,encode_dim,5,1)
    TCRencoder=load_model(model_dir)
    encoder=Model(TCRencoder.input,TCRencoder.layers[-12].output)
    return encoder.predict(TCR_contigs)