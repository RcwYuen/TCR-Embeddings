import torch
import pandas as pd
from embed.physicochemical import kidera, atchley, rand, aaprop
from embed.llm import tcrbert
from sceptr import variant
from pathlib import Path
from dotenv import load_dotenv
from reduction.autoencoder import Encoder, Autoencoder as AE
import random
import numpy as np
import warnings
import os, sys

load_dotenv()
python_path = os.getenv('PYTHONPATH')
if python_path:
    sys.path.append(python_path)

dir = Path(__file__).resolve().parent

class JohnsonLindenstarauss():
    def __init__(self, in_dim: int, fname: str = None, out_dim: int = 5):
        assert (in_dim is None) ^ (fname is None), "Ambiguous File Naming, only use `in_dim` or `fname` argument."
        self._in_dim = in_dim
        self._out_dim = out_dim
        self._fname = fname if fname is not None else f"jl-{self._in_dim}_{self._out_dim}.parquet"
        
        try:
            self.load_transformation()
        except FileNotFoundError:
            self.create_transformation()
            self.load_transformation()

    def create_transformation(self):
        transformation = np.random.normal(0, 1, size = (self._in_dim, self._out_dim))
        transformationdf = pd.DataFrame(transformation)
        transformationdf.to_parquet(dir / self._fname)

    def load_transformation(self):
        self.transformation = pd.read_parquet(dir / self._fname).to_numpy()
        self.transformation_tensor = self._to_tensor(self.transformation)

    def _to_tensor(self, ndarray):
        ndarray = torch.from_numpy(ndarray)
        ndarray.requires_grad = False
        return ndarray.cuda() if torch.cuda.is_available() else ndarray

    def reduce(self, obj):
        if type(obj) == torch.Tensor:
            transformation = self.transformation_tensor.clone()
        elif type(obj) == np.ndarray:
            transformation = self.transformation.copy()
        
        return obj @ transformation

class AutoEncoder:
    def __init__(self, encoding_method, method_name: str, out_dim = 5, batchsize: int = 1024):
        self._encoder_fname = f'{method_name}_encoder.pth'
        self._decoder_fname = f'{method_name}_decoder.pth'
        self._encoding_method = encoding_method
        self._method_name = method_name
        self._batchsize = batchsize
        self.out_dim = out_dim

        try:
            self.load_transformation()
        except FileNotFoundError:
            self._encoder = None
            warnings.warn("No AutoEncoder Transformer found for this encoding method.")

    @staticmethod
    def _batches(df, batch_size):
        for start in range(0, len(df), batch_size):
            yield df[start:start + batch_size]
    
    def create_transformation(self, dataset_paths):
        try:
            random.shuffle(dataset_paths)
            in_dim = self._encoding_method.calc_vector_representations(
                pd.read_csv(dataset_paths[0], sep = "\t", dtype = str).head()
            ).shape[1]

            autoencoder = AE(in_dim, 5)
            autoencoder = autoencoder.cuda() if torch.cuda.is_available() else autoencoder
            criterion = torch.nn.MSELoss()
            optim = torch.optim.Adam(autoencoder.parameters(), lr = 1e-4)

            for epoch in range(10):
                for fno, file in enumerate(dataset_paths):
                    df = pd.read_csv(file, sep = "\t", dtype = str)
                    df = df.sample(frac=1).reset_index(drop=True)
                    num_batches = int(np.ceil(len(df) / self._batchsize))
                    for batchno, x in enumerate(AutoEncoder.batches(df, self._batchsize)):
                        x = self._encoding_method.calc_vector_representations(x, batchsize = self._batchsize)
                        x = torch.from_numpy(x).to(torch.float32)
                        x = x.cuda() if torch.cuda.is_available() else x
                        y = autoencoder(x)
                        loss = criterion(x, y)
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        print (f"{self._method_name.upper()} | Epoch {epoch:2} / 10 - File {fno:3} / {len(dataset_paths):3} - Batch {batchno:5} / {num_batches:5} | Loss {loss.data.mean():6}")

        except KeyboardInterrupt:
            pass

        finally:
            torch.save(autoencoder.encoder.state_dict(), dir / self._encoder_fname)
            torch.save(autoencoder.decoder.state_dict(), dir / self._decoder_fname)
            self.encoder = autoencoder.encoder

    def load_transformation(self):
        in_dim = self._encoding_method.calc_vector_representations(pd.read_csv(dir / "sample.tsv", sep = "\t", dtype = str)).shape[1]
        encoder = Encoder(in_dim, 5)
        encoder.load_state_dict(torch.load(dir / self._encoder_fname))
        encoder.eval()
        self.encoder = encoder.cuda() if torch.cuda.is_available() else encoder

    def reduce(self):
        if self._encoder is None:
            raise NameError("You must train the AutoEncoder before using this function.")

        if type(obj) != torch.Tensor:
            obj = torch.from_numpy(obj).to(torch.float32)
    
        reduced = []
        for batch in AutoEncoder.batches(obj, self._batchsize):
            reduced += self._encoder(batch).tolist()
        
        return np.array(reduced)