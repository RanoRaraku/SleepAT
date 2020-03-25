"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic DPS library.
"""
import numpy as np

def delta(feats, delta_window:int):
    """
    Calculates the delta features (N,M) array.
    ** numpy can iterate over rows so maybe undo indexing
    ------------------------------------------
    Input :
        feats ....  np.ndarray(shape=(N,M)) array where N is the number of
                    observations and M is feature-vector dimension.
        window ...  number of +- neighbouring frames used to compute deltas.
                    Edge points are repeated (default: int=2.
    Output:
        feats_delta .... np.ndarray(shape=(N, M), dtype=feats.dtype)
    """
    if delta_window < 1:
        raise ValueError('Window length must be an integer >= 1')

    den = 2*sum([i**2 for i in range(1, delta_window+1)])
    feats_padded = np.pad(feats,((delta_window,delta_window),(0,0)),mode='edge')
    feats_delta = np.zeros(shape=(feats.shape),dtype=feats.dtype)
    weights = np.arange(-delta_window, delta_window+1)
    for i in np.arange(feats.shape[0]):
        feats_delta[i] = np.dot(weights,feats_padded[i : 2*delta_window+i+1]) / den
    return feats_delta