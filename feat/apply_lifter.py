"""
Made by Michal Borsky, 2019, copyright (C) RU
Acoustuc feature extraction library.
"""
import numpy as np
from sleepat.base.opts import ApplyLifterOpts

def apply_lifter(feats, config:str=None, **kwargs) -> np.ndarray:
    """
    Apply sinusoidal liftering to increase value of higher order cesptral
    coefficients.
    Input :
        feats ... a feature vector of np.ndarray(shape=(N,M)) type.
        <lifter_order> ... cepstral liftering order(default: int= <22>)
        config ...
        **kwargs ...
    Output:
        np.ndarray(shape=(N,M))
    """
    conf = ApplyLifterOpts(config,**kwargs)

    n = range(feats.shape[1])
    lift = 1 + (conf.lifter_order/2)*np.sin(n*np.pi/conf.lifter_order)
    return (feats*lift).astype(np.float32)
