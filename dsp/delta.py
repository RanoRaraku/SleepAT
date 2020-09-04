"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic DPS library.
"""
import numpy as np

def delta(feats:np.ndarray, delta_window:int) -> np.ndarray:
    """
    Calculates the delta features from an (M,N) array.
    Edge value are repeated to maintain feature dimensions.
    Input :
        feats ... features array where M is the number of
                  observations and N is feature dimension.
        delta_window ... number of +- neighbouring frames used to compute deltas.
                    Edge points are repeated (default:int=2).
    Output:
        feats_delta .... np.ndarray(shape=(M,N), dtype=feats.dtype)
    """
    if delta_window < 1:
        raise ValueError('Window length must be an integer >= 1')

    den = 2*sum([i**2 for i in range(1, delta_window+1)])
    feats_pad = np.pad(feats,((delta_window,delta_window),(0,0)),mode='edge')
    feats_delta = np.zeros(shape=(feats.shape),dtype=feats.dtype)
    weights = range(-delta_window, delta_window+1)
    for i in range(feats.shape[0]):
        feats_delta[i] = np.dot(weights,feats_pad[i : 2*delta_window+i+1])/den
    return feats_delta