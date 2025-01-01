import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from tcr_embeddings import runtime_constants
from tcr_embeddings.training import training_utils as utils
from tcr_embeddings.training.dataloader import Patients

os.chdir(runtime_constants.HOME_PATH)
sys.path.append(str(runtime_constants.HOME_PATH))


def compute_auc_stats(pth: Path) -> list[tuple[int, float]]:
    ls_aucs: list[tuple[int, float]] = []

    for epoch in pth.glob("Epoch */test-records.csv"):
        df_test_records = pd.read_csv(epoch)
        current_epoch = int(epoch.parent.name.replace("Epoch ", ""))
        auc_score = roc_auc_score(df_test_records["actual"], df_test_records["pred"])
        ls_aucs.append((current_epoch, auc_score))

    return ls_aucs


def find_best_epoch(pth: Path) -> tuple[int, float]:
    return _find_best_epoch(compute_auc_stats(pth))


def _find_best_epoch(ls_aucs: list[tuple[int, float]]) -> tuple[int, float]:
    """
    Priority is given to high AUC > high epoch counts to avoid premature fitting.

    This is created to help unittests.
    """
    return max(ls_aucs[::-1], key=lambda x: x[1])


def load_trained_model(
    model: torch.nn.Module, best_epoch: int, outpath: Path
) -> torch.nn.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.load_state_dict(
            torch.load(outpath / f"Epoch {best_epoch}/classifier.pth")
        )

    for param in model.parameters():
        param.requires_grad = False
    return model


def run(
    configs: dict, dataset: Patients, model: torch.nn.Module, outpath: Path
) -> None:
    dataset.test()
    best_epoch, best_epoch_auc = find_best_epoch(outpath)
    best_model = load_trained_model(model, best_epoch, outpath)
    outpath = utils.make_directory_where_necessary(outpath / "interpretability")
    dataset.set_get_fname_mode(True)
    dct_interpretability_record: dict[str, list[float | str | Path]] = {
        "filename": [],
        "true": [],
        "prediction": [],
    }

    utils.printf(f"Used Epoch: {best_epoch}; AUC: {best_epoch_auc}", "INFO")

    best_model.eval()
    with torch.no_grad():
        label: int
        fname: Path
        df: pd.DataFrame
        for idx, (label, fname, df) in enumerate(dataset):  # type: ignore
            utils.printf(f"Processing File {idx} / {len(dataset)}.  True Label {label}")
            pred = best_model(
                utils.calculate_embeddings(df, configs, dataset.get_ext())
            ).item()
            utils.printf(f"File {idx} / {len(dataset)}: Predicted Value: {pred}")

            dct_interpretability_record["filename"].append(fname)
            dct_interpretability_record["true"].append(label)
            dct_interpretability_record["prediction"].append(pred)

            if int(pred > 0.5) == label:
                utils.printf(
                    f"File {idx} / {len(dataset)}: Correct Prediction, finding non-zero TCRs"
                )
                ls_nonzero_idx = torch.nonzero(best_model.last_weights)[:, 0]
                ls_ws = best_model.last_weights[ls_nonzero_idx][:, 0].tolist()
                df_nonzero_idx = (
                    pd.read_csv(utils.get_original_path(fname), sep="\t", dtype=str)
                    if dataset.get_ext() == ".pq"
                    else df.copy()
                )
                df_nonzero_idx = df_nonzero_idx.iloc[ls_nonzero_idx.tolist()].copy()
                df_nonzero_idx["assigned_weights"] = ls_ws
                df_nonzero_idx.infer_objects().to_parquet(
                    outpath / fname.name.replace(".tsv", ".pq")
                )

            else:
                utils.printf(f"File {idx} / {len(dataset)}: Incorrect Prediction.")

    pd.DataFrame(dct_interpretability_record).to_csv(outpath / "eval.csv", index=False)
