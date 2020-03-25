"""
Made by Michal Borsky, 2019, copyright (C) RU
Acoustuc feature extraction library.
"""
import numpy as np
from sleepat.base.opts import SpliceFramesOpts

def splice_frames(feats, config:str=None, **kwargs) -> np.ndarray:
    """
    Apply across temporal splicing. The output

    Input :
        feats ... a feature vector of np.ndarray(shape=(N,M)) type.
        <context_left> ... number of left frames to splice (default: int= <4>)
        <context_right> ... number of right frames to splice (default: int= <4>)
        config ...
        **kwargs ...
    Output:
        np.ndarray(shape=(N, M*(left+right+1)))
    """
    conf = SpliceFramesOpts(config,**kwargs)

    if (conf.context_left < 0 ) or (conf.context_right < 0):
        raise ValueError('Left and right context must be >= 0')

    (N, M) = feats.shape
    feats_left = np.zeros(shape=(N, M*conf.context_left),dtype=feats.dtype)
    feats_right = np.zeros(shape=(N, M*conf.context_right),dtype=feats.dtype)
    feats_padded = np.pad(feats,((conf.context_left, conf.context_right),(0,0)),mode='edge')
    for n in range(N):
        feats_left[n] = feats_padded[n : n+conf.context_left].flatten()
        feats_right[n] = feats_padded[n+conf.context_left+1 : n+conf.context_left+conf.context_right+1].flatten()
    return np.concatenate((feats_left,feats,feats_right), axis=1).astype(np.float32)
