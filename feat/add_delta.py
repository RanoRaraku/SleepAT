"""
Made by Michal Borsky, 2019, copyright (C) RU
Acoustuc feature extraction library.
"""
import numpy as np
import sleepat
from sleepat import dsp, opts

def add_delta(feats:np.ndarray, config:str=None, **kwargs) -> np.ndarray:
    """
    Adds dynamic features to feature array. Edge values are repeated
    to maintain feature dimension. Dynamic features are concatenated
    in the ascending order along the x-axis after static features.
    Optional arguments can be passed in a config file or as kwargs.

    Arguments:
        feats ... array of values of np.ndarray(shape=(m,n)) shape
        <delta_order> ... order of delta coefficients (default:int = 2)
        <delta_window> ... no +-context frames to delta computation (default:int = 2)
        config ... configuration file with optional arguments (default:str=None)
        **kwargs ... optional arguments passed as kwargs

        out ... array with static and dynamic features, shape=(m,(n+1)*order)
    """
    conf = opts.AddDelta(config,**kwargs)
    if conf.delta_order < 1:
        raise ValueError('Delta order must be an integer >= 1')
    (m,n) = feats.shape

    out = np.empty(shape=(m,n*(conf.delta_order+1)),dtype=feats.dtype)
    out[0:m,0:n] = feats
    for j in range(1, conf.delta_order+1):
        dynamic = dsp.delta(out[:,n*(j-1):n*j], conf.delta_window)
        out[0:m,n*j:n*(j+1)] = dynamic
    return out
