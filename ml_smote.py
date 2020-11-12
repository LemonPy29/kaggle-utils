import pandas as pd
import numpy as np
import sklearn
import cudf 
import cupy
import cuml
from functools import wraps


def tail_mask(y):
    irpl = y.sum(axis=0)
    irpl = irpl.max() / irpl
    mir = irpl.mean()
    return y[y.columns[values(irpl > mir)]].any(axis=1) == 1

def convex_comb(p1, p2, nu):
    return p1 + nu * (p2 - p1)

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
    try:
        return x.values
    except NotImplementedError:
        return x.values_host

class MLSmote:
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

    @as_np_array
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

    def _resample_features(self, X, sample_idx, nbs_idx, categorical):
        if not categorical: 
            categorical = []
        X_num = X.drop(categorical, axis=1).values
        cat_values = values(X[categorical])
        nu = np.random.uniform(0, 1, len(sample_idx))
        X_res = convex_comb(X_num[sample_idx], X_num[nbs_idx]), nu)
        cat_values = self.arrayb.array(
                np.where(nu < .5, 
                         cat_values[sample_idx],
                         cat_values[nbs_idx])     
                )
        
        X_res = self.arrayb.concatenate(cat_values, X_res, axis=1)
        return pd.DataFrame(X_res, columns=X.columns)

    def resample(self, X, y, n_samples, categorical=None):
        np.random.seed(self.seed)
        mask = tail_mask(y)
        X_masked = X[mask]
        y_masked = y[mask]
        idxs = self.nn_wrapper(X_masked)
        sample_idx = np.random.choice(idxs[:, 0], n_samples)
        nbs_idx = self.arrayb.array([np.random.choice(idxs[j, 1:]) for j in sample_idx])

        X_res = _resample_features(X_masked, sample_idx, nbs_idx,
                                   categorical=categorical)

        nn_sum = self.nn_sum(y_masked, idxs[sample_idx])
        y_res = self.dfb.DataFrame(
            self.arrayb.where(nn_sum > 2, 1, 0),
            columns=y.columns
            )

        return X_res, y_res
