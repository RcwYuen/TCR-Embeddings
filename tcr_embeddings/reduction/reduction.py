import os
import random
import sys
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from tcr_embeddings import runtime_constants

os.chdir(runtime_constants.HOME_PATH)
sys.path.append(str(runtime_constants.HOME_PATH))


class NoReduce:
    def __init__(self):
        pass

    def reduce(self, obj: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return obj / (np.linalg.norm(obj, axis=1)[:, None])


class JohnsonLindenstarauss:
    def __init__(self, in_dim: int, fname: str | None = None, out_dim: int = 5):
        assert (in_dim is None) ^ (
            fname is None
        ), "Ambiguous File Naming, only use `in_dim` or `fname` argument."

        self._in_dim = in_dim
        self._out_dim = out_dim
        self._fname = (
            fname if fname is not None else f"jl-{self._in_dim}_{self._out_dim}.parquet"
        )

        try:
            self.load_transformation()
        except FileNotFoundError:
            self.create_transformation()
            self.load_transformation()

    def create_transformation(self) -> None:
        df_transformation = pd.DataFrame(
            np.random.normal(0, 1, size=(self._in_dim, self._out_dim))
        )
        df_transformation.to_parquet(
            Path(__file__).resolve().parent / "jl-transformations" / self._fname
        )

    def load_transformation(self) -> None:
        self.transformation: np.ndarray = pd.read_parquet(
            Path(__file__).resolve().parent / "jl-transformations" / self._fname
        ).to_numpy()
        self.transformation_tensor: torch.Tensor = self._to_tensor(self.transformation)

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(arr)
        tensor.requires_grad = False
        return tensor.cuda() if torch.cuda.is_available() else tensor

    def reduce(self, obj: torch.Tensor | np.ndarray):
        transformation: np.ndarray | torch.Tensor
        transformed: np.ndarray | torch.Tensor

        match type(obj):
            case torch.Tensor:
                # mypy ignoring below as we checked the type is torch.Tensor before going in, so the following
                # operation is safe
                obj = obj.cuda() if torch.cuda.is_available() else obj # type: ignore
                transformation = self.transformation_tensor.clone()
                transformed = obj @ transformation
                norms = torch.norm(transformed, dim=1).unsqueeze(1)

            case np.ndarray:
                transformation = self.transformation.copy()
                transformed = obj @ transformation
                norms = np.linalg.norm(transformed, axis=1)[:, None]

            case _:
                raise TypeError(
                    "Expected either torch.Tensor datatype or np.ndarray DataType"
                )

        return transformed / norms


class _Encoder(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(_Encoder, self).__init__()
        self._encoder = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._encoder(x)


class _Decoder(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(_Decoder, self).__init__()
        self._decoder = torch.nn.Linear(out_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._decoder(x)


class _Autoencoder(torch.nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super(_Autoencoder, self).__init__()
        self.encoder = _Encoder(in_dim, latent_dim)
        self.decoder = _Decoder(in_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoEncoder:
    def __init__(
        self,
        encoding_method: Callable,
        method_name: str,
        out_dim=5,
        batchsize: int = 1024,
    ):
        self._encoder_fname = f"{method_name}_encoder.pth"
        self._decoder_fname = f"{method_name}_decoder.pth"
        self._encoding_method = encoding_method
        self._method_name = method_name
        self._batchsize = batchsize
        self.out_dim = out_dim
        self._enforce_norm = "sceptr" in method_name

        assert hasattr(
            self._encoding_method, "calc_vector_representations"
        ), "Encoding Method passed is not an eligible encoding method."

        try:
            self.load_transformation()
        except FileNotFoundError:
            self.encoder: None | _Encoder = None
            warnings.warn("No AutoEncoder Transformer found for this encoding method.")

    @staticmethod
    def _batches(df, batch_size):
        for start in range(0, len(df), batch_size):
            yield df[start : start + batch_size]

    def get_in_dim(self):
        return self._encoding_method.calc_vector_representations(  # type: ignore
            pd.read_csv(runtime_constants.DATA_PATH / "sample.tsv", sep="\t", dtype=str)
        ).shape[1]

    def create_transformation(self, dataset_paths):
        try:
            random.shuffle(dataset_paths)
            in_dim = self._encoding_method.calc_vector_representations(  # type: ignore
                pd.read_csv(dataset_paths[0], sep="\t", dtype=str).head()
            ).shape[1]

            autoencoder = _Autoencoder(in_dim, 5)
            autoencoder = (
                autoencoder.cuda() if torch.cuda.is_available() else autoencoder
            )
            criterion = torch.nn.MSELoss()
            optim = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

            for epoch in range(10):
                for fno, file in enumerate(dataset_paths):
                    df = pd.read_csv(file, sep="\t", dtype=str)
                    df = df.sample(frac=1).reset_index(drop=True)
                    num_batches = int(np.ceil(len(df) / self._batchsize))
                    for batchno, x in enumerate(
                        AutoEncoder._batches(df, self._batchsize)
                    ):
                        try:
                            x = self._encoding_method.calc_vector_representations(  # type: ignore
                                x, batchsize=self._batchsize
                            )
                        except TypeError:
                            x = self._encoding_method.calc_vector_representations(x)  # type: ignore

                        x = torch.from_numpy(x).to(torch.float32)
                        x = x.cuda() if torch.cuda.is_available() else x
                        y = autoencoder(x)
                        loss = criterion(x, y)
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        print(
                            f"{self._method_name.upper()} | Epoch {epoch:2} / 10 - File {fno:3} / {len(dataset_paths):3} - Batch {batchno:5} / {num_batches:5} | Loss {loss.data.mean():6}"
                        )

        except KeyboardInterrupt:
            pass

        finally:
            torch.save(
                autoencoder.encoder.state_dict(),
                Path(__file__).resolve().parent
                / "ae-transformations"
                / self._encoder_fname,
            )
            torch.save(
                autoencoder.decoder.state_dict(),
                Path(__file__).resolve().parent
                / "ae-transformations"
                / self._decoder_fname,
            )
            self.encoder = autoencoder.encoder

    def load_transformation(self):
        in_dim = self._encoding_method.calc_vector_representations(  # type: ignore
            pd.read_csv(runtime_constants.DATA_PATH / "sample.tsv", sep="\t", dtype=str)
        ).shape[1]

        encoder = _Encoder(in_dim, 5)

        if torch.cuda.is_available():
            encoder.load_state_dict(
                torch.load(
                    Path(__file__).resolve().parent
                    / "ae-transformations"
                    / self._encoder_fname
                )
            )
        else:
            encoder.load_state_dict(
                torch.load(
                    Path(__file__).resolve().parent
                    / "ae-transformations"
                    / self._encoder_fname,
                    map_location=torch.device("cpu"),
                )
            )

        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

        self.encoder = encoder.cuda() if torch.cuda.is_available() else encoder

    def reduce(self, obj):
        if self.encoder is None:
            raise NameError(
                "You must train the AutoEncoder before using this function."
            )

        match type(obj):
            case np.ndarray:
                obj = torch.from_numpy(obj).to(torch.float32)

            case torch.Tensor:
                pass

            case _:
                raise TypeError("Expected only np.ndarray or torch.Tensor")

        obj = obj.cuda() if torch.cuda.is_available() else obj
        obj = obj.float()
        reduced: list | np.ndarray = []
        for batch in AutoEncoder._batches(obj, self._batchsize):
            reduced += self.encoder(batch).tolist()
        reduced = np.array(reduced)

        return reduced / (np.linalg.norm(reduced, axis=1)[:, None])
