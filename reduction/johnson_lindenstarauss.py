import numpy as np
import pandas as pd
from pathlib import Path
import torch
import argparse
import warnings

dir = Path(__file__).resolve().parent

def create_transformation(in_dim, out_dim = 5, fname = None, to_tensor = False):
    fname = fname if fname is not None else f"jl-{in_dim}_{out_dim}.parquet"
    transformation = np.random.normal(0, 1, size = (in_dim, out_dim))
    transformationdf = pd.DataFrame(transformation)
    transformationdf.to_parquet(dir / fname)
    return transformation if not to_tensor else _to_tensor(transformation)

def load_transformation(in_dim = None, fname = None, out_dim = 5, to_tensor = False):
    assert (in_dim is None) ^ (fname is None), "Ambiguous File Naming, only use `in_dim` or `fname` argument."
    fname = fname if fname is not None else f"jl-{in_dim}_{out_dim}.parquet"
    transformation = pd.read_parquet(dir / fname).to_numpy()
    return transformation if not to_tensor else _to_tensor(transformation)
    
def _to_tensor(ndarray):
    ndarray = torch.from_numpy(ndarray)
    ndarray.requires_grad = False
    return ndarray.cuda() if torch.cuda.is_available() else ndarray

def parse_cmd_args():
    parser = argparse.ArgumentParser(
        usage = "Creates Linear Transformation Parquet Files for the application of the Johnson Lindenstarauss Lemma"
    )
    parser.add_argument(
        "dim", help = "Input Dimension of the High Dimensional Vector Space"
    )
    parser.add_argument(
        "-f", "--fname", help = "Specifying Filename for the output parquet file.  Not recommended."
    )
    return parser.parse_args()

def _ndarray_reduce(ndarray, out_dim = 5):
    transformation = load_transformation(in_dim = ndarray.shape[1], out_dim = out_dim)
    return ndarray @ transformation

def _tensor_reduce(tensor, out_dim = 5):
    transformation = load_transformation(in_dim = tensor.shape[1], out_dim = out_dim, to_tensor = True)
    return tensor @ transformation

def reduce(obj, out_dim = 5, batchsize = 2 ** 10, encoder = None):
    if type(obj) == torch.Tensor:
        return _tensor_reduce(obj, out_dim=out_dim)
    elif type(obj) == np.ndarray:
        return _ndarray_reduce(obj, out_dim=out_dim)

if __name__ == "__main__":
    args = parse_cmd_args()
    
    if args.fname is not None:
        warnings.warn("It is not recommended to specify a filename")

    create_transformation(
        in_dim = int(args.dim),
        fname = args.fname
    )