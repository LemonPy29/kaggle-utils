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
            self.dfb = cudf
            self.arrayb = cupy
            self.mlb = cuml
        elif backend == 'cpu':
            self.dfb = pd
            self.arrayb = np
            self.mlb = sklearn
        self.nn_args = nn_args
        self.seed = seed

    def nn_wrapper(self, X):
        nn = self.mlb.neighbors.NearestNeighbors(**self.nn_args)
        nn.fit(X)
        _, idxs = nn.kneighbors(X)
        return idxs

    def nn_sum(self, y, all_nbs):
        res = self.arrayb.zeros((len(all_nbs), y.shape[1]))
        for i, nbs in enumerate(all_nbs):
            res[i] = y[nbs].sum(axis=0)
        return res

    def resample(self, X, y, n_samples):
        np.random.seed(self.seed)
        mask = tail_mask(y)
        X_masked = X[mask].reset_index(drop=True)
        y_masked = y[mask].reset_index(drop=True)
        idxs = values(self.nn_wrapper(X_masked))
        
        sample_idx = np.random.choice(idxs[:, 0], n_samples)
        nbs_idx = [np.random.choice(idxs[j, 1:]) for j in sample_idx]
        nn_sum = self.nn_sum(y_masked.values, idxs[sample_idx])
        y_res = self.dfb.DataFrame(
            self.arrayb.where(nn_sum > 2, 1, 0),
            columns=y.columns
            )
        X_res = inbetween_sample(X_masked.loc[sample_idx],
                                X_masked.loc[nbs_idx])
        
        return self.dfb.concat([X, X_res], axis=0), self.dfb.concat([y, y_res], axis=0)
