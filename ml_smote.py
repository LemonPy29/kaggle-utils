import pandas as pd
import numpy as np
import sklearn
import cudf 
import cupy
import cuml
from functools import wraps


def tail_mask(y, minority_class=1):
    irpl = np.zeros(len(y.columns))
    for i, col in enumerate(y.columns):
        irpl[i] = y[col].value_counts()[minority_class]
    irpl = irpl.max() / irpl
    mir = irpl.mean()
    return y[y.columns[(irpl > mir)]].any(axis=1) == 1

def inbetween_sample(p1, p2):
    nu = np.random.uniform(0, 1)
    return p1 + nu * (p1 - p2)

def as_np_array(fn):
    @wraps(as_np_array)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if isinstance(result, cupy.ndarray):
            result = result.get()
        return result
    return wrapper

@as_np_array
def values(x):
    return x.values

class MLSMOTE:
    def __init__(self, *, backend='cuda', seed=123, **nn_args):
        if backend == 'cuda':
            self.db = cudf
            self.cb = cuml
        elif backend == 'cpu':
            self.db = pd
            self.cb = sklearn
        self.nn_args = nn_args
        self.seed = seed

    def nn_wrapper(self, X):
        nn = self.cb.neighbors.NearestNeighbors(**self.nn_args)
        nn.fit(X)
        _, idxs = nn.kneighbors(X)
        return idxs

    def resample(self, X, y, n_samples):
        mask = tail_mask(y)
        X_masked = X[mask].reset_index(drop=True)
        y_masked = y[mask].reset_index(drop=True)
        idxs = values(self.nn_wrapper(X_masked))
        X_res, y_res = [], []
        
        for i in range(n_samples):
            np.random.seed(i + self.seed)
            sample_idx = np.choice(idxs[:, 1])
            nbs_idx = np.choice(idxs[sample_idx, 1:])
            ser = y.loc[idxs[sample_idx]].sum(axis=0)
            ynew = y_res.append(ser.mask(ser > 2, 1, 0))
            xnew = inbetween_sample(X_masked.loc[sample_idx],
                                    X_masked.loc[nbs_idx])
            X_res.apppend(xnew)
        
        return (self.db.concat(X, self.db.DataFrame(X_res), axis=1),
                self.db.concat(y, self.db.DataFrame(y_res), axis=1))
