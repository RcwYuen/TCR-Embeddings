import torch
import pandas as pd
from embed.physicochemical import kidera, atchley, rand, aaprop
from embed.llm import tcrbert
from sceptr import variant
from pathlib import Path
import random
import numpy as np

dir = Path(__file__).resolve().parent

class Encoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Encoder, self).__init__()
        self._encoder = torch.nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return self._encoder(x)
    
class Decoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Decoder, self).__init__()
        self._decoder = torch.nn.Linear(out_dim, in_dim)
    
    def forward(self, x):
        return self._decoder(x)

class Autoencoder(torch.nn.Module):
    def __init__(self, in_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(in_dim, latent_dim)
        self.decoder = Decoder(in_dim, latent_dim)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def batches(df, batch_size):
    for start in range(0, len(df), batch_size):
        yield df[start:start + batch_size]

def train(dataset_paths, encoding_method, method_name, batchsize = 2 ** 10):
    try:
        random.shuffle(dataset_paths)
        in_dim = encoding_method.calc_vector_representations(
            pd.read_csv(dataset_paths[0], sep = "\t", dtype = str).head()
        ).shape[1]

        autoencoder = Autoencoder(in_dim, 5)
        autoencoder = autoencoder.cuda() if torch.cuda.is_available() else autoencoder
        criterion = torch.nn.MSELoss()
        optim = torch.optim.Adam(autoencoder.parameters(), lr = 1e-3, weight_decay = 0.01)

        for epoch in range(10):
            for fno, file in enumerate(dataset_paths):
                df = pd.read_csv(file, sep = "\t", dtype = str)
                df = df.sample(frac=1).reset_index(drop=True)
                num_batches = int(np.ceil(len(df) / batchsize))
                for batchno, x in enumerate(batches(df, batchsize)):
                    x = encoding_method.calc_vector_representations(x, batchsize = batchsize)
                    x = torch.from_numpy(x).to(torch.float32)
                    x = x.cuda() if torch.cuda.is_available() else x
                    y = autoencoder(x)
                    loss = criterion(x, y)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    print (f"{encoding_method.upper()} | Epoch {epoch:2} / 10 - File {fno:3} / {len(dataset_paths):3} - Batch {batchno:5} / {num_batches:5} | Loss {loss.data.mean():6}")
    
    except KeyboardInterrupt:
        pass

    finally:
        torch.save(autoencoder.encoder.state_dict(), dir / f'{method_name}_encoder.pth')
        torch.save(autoencoder.decoder.state_dict(), dir / f'{method_name}_decoder.pth')

def load(encoding_method, method_name):
    in_dim = encoding_method.calc_vector_representations(pd.read_csv(dir / "sample.tsv", sep = "\t", dtype = str)).shape[1]
    encoder = Encoder(in_dim, 5)
    encoder.load_state_dict(torch.load(dir / f"{method_name}_encoder.pth"))
    encoder.eval()
    encoder = encoder.cuda() if torch.cuda.is_available() else encoder
    return encoder

def reduce(obj, encoder = None, out_dim = 5, batchsize = 2 ** 10):
    assert encoder is not None, "Please provide an Encoder"

    if type(obj) != torch.Tensor:
        obj = torch.from_numpy(obj).to(torch.float32)
    
    reduced = []
    for batch in batches(obj, batchsize):
        reduced += encoder(batch).tolist()
    
    return np.array(reduced)

if __name__ == "__main__":
    dataset_paths = sum(
        [list((Path.cwd() / "data/tcvhcw/cleaned").glob("*.tsv")),
        list((Path.cwd() / "data/Tx/cleaned").glob("*.tsv"))], []
    )
    train(dataset_paths, atchley, "atchley")
    train(dataset_paths, aaprop, "aaprop")
    train(dataset_paths, rand, "rand")
    train(dataset_paths, kidera, "kidera")
    train(dataset_paths, tcrbert, "tcrbert")
    
    sceptr_default = variant.default()
    train(dataset_paths, sceptr_default, "sceptr-default")

    sceptr_tiny = variant.tiny()
    train(dataset_paths, sceptr_tiny, "sceptr_tiny")