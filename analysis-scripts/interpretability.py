import sys
sys.path.append("C:/Users/rcwyuen/OneDrive/Studies/UCL/Publications/TCR-Embeddings")

from pathlib import Path
from sklearn.metrics import roc_auc_score
import pandas as pd
import training.models as model
import reduction.reduction as reducer
import torch
import os
from tqdm import tqdm

data_src = "results/"
src = Path("C:/Users/rcwyuen/OneDrive/Studies/UCL/Publications/TCR-Embeddings")

def read_kfold_set(pth, label):
    with open(pth, "r") as f:
        kf = f.readlines()
        assert len(kf) == 1
        return [(file, label) for file in kf[0].split("<>")]

def find_best_epoch(pth):
    '''
    Priority is given to high AUC > high epoch counts.
    '''
    ls_aucs = []

    for epoch in pth.glob("Epoch */test-records.csv"):
        df_test_records = pd.read_csv(epoch)
        ls_aucs.append((
            int(epoch.parent.name.replace("Epoch ", "")),
            roc_auc_score(df_test_records["actual"], df_test_records["pred"])
        ))

    return max(ls_aucs[::-1], key = lambda x: x[1])

def method_name_to_func(method_name):
    if method_name == "atchley":
        from embed.physicochemical import atchley
        return atchley()
        
    elif method_name == "kidera":
        from embed.physicochemical import kidera
        return kidera()
        
    elif method_name == "rand":
        from embed.physicochemical import rand
        return rand()
        
    elif method_name == "aaprop":
        from embed.physicochemical import aaprop
        return aaprop()
        
    elif method_name == "tcrbert":
        from embed.llm import tcrbert
        return tcrbert()
        
    elif method_name == "sceptr-tiny":
        from sceptr import variant
        return variant.tiny()
        
    elif method_name == "sceptr-default":
        from sceptr import variant
        return variant.default()
    
    else:
        raise ValueError("Cannot parse Method Name.")

def find_method(pth):
    if "autoencoder" in pth.parent.name:
        encoding_method_str = pth.parent.name.replace(f"-autoencoder", "")
        encoding_method = method_name_to_func(encoding_method_str)
        reduction_method = reducer.AutoEncoder(
            encoding_method, encoding_method_str
        )

    elif "johnson-lindenstarauss" in pth.parent.name:
        encoding_method_str = pth.parent.name.replace(f"-johnson-lindenstarauss", "")
        encoding_method = method_name_to_func(encoding_method_str)
        reduction_method = reducer.JohnsonLindenstarauss(
            encoding_method.calc_vector_representations(
                pd.read_csv(Path.cwd() / "data/sample.tsv", sep = "\t", dtype = str)
            ).shape[1]
        )

    elif "no-reduction" in pth.parent.name:
        encoding_method_str = pth.parent.name.replace(f"-no-reduction", "")
        encoding_method = method_name_to_func(encoding_method_str)
        reduction_method = reducer.NoReduce()

    else:
        raise ValueError("Cannot parse method.")
    
    return (encoding_method, reduction_method)

def load_trained_model(model_encoding, model_reducer):
    if isinstance(model_reducer, reducer.NoReduce):
        model_trained = model.ordinary_classifier(model_encoding, True)
    else:
        model_trained = model.reduced_classifier(True)
    
    model_trained.load_state_dict(
        torch.load(method_kf / f"Epoch {best_epoch}/classifier.pth")
    )

    for param in model_trained.parameters():
        param.requires_grad = False
    
    return model_trained

def get_result(pth_positive_file):
    df = pd.read_csv(src / pth_positive_file, sep="\t", dtype=str)

    # Embedding File
    tensor_embeddings = model_encoding.calc_vector_representations(df)
    
    # Reducing Dimensionality
    tensor_embeddings = model_reducer.reduce(tensor_embeddings)
    
    # Tensoring
    tensor_embeddings = torch.from_numpy(tensor_embeddings).to(torch.float32)
    tensor_embeddings = tensor_embeddings.cuda() if torch.cuda.is_available() else tensor_embeddings
    
    # Creating Prediction
    predicted_label = model_trained(tensor_embeddings)

    # Non-Zero Weights
    nonzero_idx = torch.nonzero(model_trained.last_weights)[:, 0]
    ls_ws = model_trained.last_weights[nonzero_idx][:, 0].tolist()
    df = df.iloc[nonzero_idx.tolist()]
    df["assigned_weights"] = ls_ws

    return df, predicted_label

def make_interpretability_dir():
    try:
        os.makedirs(method_kf / "interpretability")
    except:
        pass

ls_method_kf_dirs = list((Path.cwd() / data_src).glob("**/kfold-*"))

for idx, method_kf in enumerate(ls_method_kf_dirs):
    ls_positive = read_kfold_set(method_kf / "pos0-kfold.txt", 1)
    ls_negative = read_kfold_set(method_kf / "neg0-kfold.txt", 0)
    ls_kfs = ls_positive + ls_negative
    best_epoch, best_epoch_auc = find_best_epoch(method_kf)
    model_encoding, model_reducer = find_method(method_kf)
    model_trained = load_trained_model(model_encoding, model_reducer)    
    make_interpretability_dir()

    df_repertoire_with_label = pd.DataFrame({"filenames": [], "true": [], "prediction": []})
    for pth_file, label in tqdm(ls_kfs, desc=f"{method_kf.name}; Progress: {idx+1}/{len(ls_method_kf_dirs)}"):
        df, pred = get_result(pth_file)
        df_repertoire_with_label = pd.concat([
            df_repertoire_with_label,
            pd.DataFrame({"filenames": [pth_file], "true": [label], "prediction": [pred.item()]})
        ])
        df.to_parquet(
            method_kf / "interpretability" / Path(pth_file).name.replace(".tsv", ".pq")
        )

    df_repertoire_with_label.set_index("filenames").to_csv(method_kf / "interpretability/results_log.csv")

    with open(method_kf / "interpretability/used_model.txt", "w") as f:
        f.write(f"Used Epoch: {best_epoch} with AUC {best_epoch_auc}")