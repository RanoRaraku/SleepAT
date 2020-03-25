"""
Made by Michal Borsky, 2019, copyright (C) RU
Acoustuc feature extraction library.
"""
import numpy as np
from sleepat.base.opts import ApplyMvnOpts
from sleepat.io import read_npy

def apply_mvn(feats, stats_file:str=None, config:str=None, **kwargs) -> np.ndarray:
    """
    Apply means and variance normalization. Can be on per-speaker or per-utterance
    basis. Per-utterance is done when stats_file=None, otherwise a proper stats npy
    file is needed.
    Input :
        feats ... a feature vector of np.ndarray(shape=(N,M)) type.
        norm_vars .... normalize variance to 1 (default:bool = False)
    Output:
        np.ndarray(shape=(N, M), dtype=feats.dtype)
    """
    conf = ApplyMvnOpts(config,**kwargs)

    if stats_file is None:
        mu = feats.mean(axis=0)
        sigma = feats.std(axis=0)
    else:
        stats = read_npy(stats_file)
        mu = stats[0]
        sigma = stats[1]

    feats -= mu
    if conf.norm_vars:
        feats /= np.sqrt(sigma + np.finfo(float).eps)
    return feats.astype(np.float32)
