import argparse
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import tqdm
from sceptr import Sceptr

from tcr_embeddings import runtime_constants
from tcr_embeddings.embed.embedder import Embedder
from tcr_embeddings.embed.physicochemical import PhysicoChemicalEncoder
from tcr_embeddings.training import training_utils as utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--encoding", help="Encoding Method")
    return parser.parse_args()


def embed_and_save(
    encoding: str,
    embedder: Sceptr | Embedder,
    filepath: Path,
    df: pd.DataFrame,
    pbar: tqdm.std.tqdm,
) -> None:
    savepath = utils.make_directory_where_necessary(
        filepath.parent.parent / runtime_constants.PRE_EMBED_PATH / encoding
    )
    save_to = savepath / filepath.name.replace(dataset.get_ext(), ".pq")
    if not save_to.exists():
        pd.DataFrame(embedder.calc_vector_representations(df)).to_parquet(save_to)
    pbar.update(1)


if __name__ == "__main__":
    parser = parse_args()
    encoding = parser.encoding.lower()
    logger = utils.create_logger(runtime_constants.HOME_PATH, "embedall.log")

    assert (
        encoding in runtime_constants.LS_VALID_ENCODING
    ), "Please input a valid encoding method."
    dataset = utils.create_dataset({"kfold": 0}, verbose=False)
    dataset.set_get_fname_mode(True)
    pbar = tqdm.tqdm(total=dataset.total_files())
    print("")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedder = utils.get_embedding_method({"encoding": encoding})

    if isinstance(embedder, PhysicoChemicalEncoder):
        print("Using multi-threaded processing...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            dataset.train()
            for label, filepath, df in dataset:  # type: ignore
                executor.submit(embed_and_save, encoding, embedder, filepath, df, pbar)

            dataset.validation()
            for label, filepath, df in dataset:  # type: ignore
                executor.submit(embed_and_save, encoding, embedder, filepath, df, pbar)

            dataset.test()
            for label, filepath, df in dataset:  # type: ignore
                executor.submit(embed_and_save, encoding, embedder, filepath, df, pbar)

    else:
        print("Using single-threaded processing...")
        dataset.train()
        for label, filepath, df in dataset:  # type: ignore
            embed_and_save(encoding, embedder, filepath, df, pbar)

        dataset.validation()
        for label, filepath, df in dataset:  # type: ignore
            embed_and_save(encoding, embedder, filepath, df, pbar)

        dataset.test()
        for label, filepath, df in dataset:  # type: ignore
            embed_and_save(encoding, embedder, filepath, df, pbar)

    for pth in (
        runtime_constants.PATH_NEGATIVE_CLASS + runtime_constants.PATH_POSITIVE_CLASS
    ):
        with open(runtime_constants.HOME_PATH / pth / "kfold.txt") as original_file:
            savepath = utils.make_directory_where_necessary(
                (runtime_constants.HOME_PATH / pth).parent
                / runtime_constants.PRE_EMBED_PATH
                / encoding
            )
            with open(savepath / "kfold.txt", "w") as savefile:
                savefile.writelines(original_file.read().replace(".tsv", ".pq"))

    logger.close()
    os.remove("embedall.log")
