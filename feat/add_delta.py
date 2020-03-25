"""
Made by Michal Borsky, 2019, copyright (C) RU
Acoustuc feature extraction library.
"""
import numpy as np
from sleepat.base.opts import AddDeltaOpts
from sleepat.dsp import delta

def add_delta(feats, config:str=None, **kwargs) -> np.ndarray:
    """
    Adds dynamic features to feature array.
    ---------------------------------------
    Input :
        feats ... a feature vector of np.ndarray(shape=(N,M)) type.
        <delta_order> ... order of delta coefficients (default:int = 2)
        <delta_window> ... no +-context frames to delta computation (default:int = 2)
        config ...
        **kwargs ...
    Output:
        np.ndarray(shape=(N,(M+1)*order))
    """
    conf = AddDeltaOpts(config,**kwargs)

    if conf.delta_order < 1:
        raise ValueError('Order must be an integer >= 1')

    M = feats.shape[1]
    for j in range(1, conf.delta_order+1):
        dynamic = delta(feats[:,(j-1)*M : j*M], conf.delta_window)
        feats = np.concatenate((feats, dynamic), axis = 1)
    return feats.astype(np.float32)
