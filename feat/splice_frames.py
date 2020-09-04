"""
Made by Michal Borsky, 2019, copyright (C) RU
Acoustuc feature extraction library.
"""
import numpy as np
import sleepat
from sleepat import opts

def splice_frames(feats, config:str=None, **kwargs) -> np.ndarray:
    """
    Splice neighbouring frames as new coefficients to current frame.
    The spliced features are added to the original vector in the
    following order [left,orig,right]. An alternative to adding delta
    features. Optional arguments <> can be defined in a config file
    or as kwargs.
    Input :
        feats ... a feature vector of np.ndarray(shape=(N,M)) type.
        <splice_left> ... number of left frames to splice (default: int= <4>)
        <splice_right> ... number of right frames to splice (default: int= <4>)
        config ... configuration files with optional arguments (default:str = None)
        **kwargs ... optional arguments as kwargs
    Output:
        np.ndarray(shape=(N, M*(left+right+1)))
    """
    conf = opts.SpliceFrames(config,**kwargs)
    (M, N) = feats.shape
    if (conf.splice_left < 0 ) or (conf.splice_right < 0):
        raise ValueError('Error: Left and right context must be >= 0')
    if (conf.splice_left+conf.splice_right) >= M:
        print(f'Error: Left + Right context >= to feature size({M}).')
        exit(1)

    feats_left = np.empty(shape=(M,N*conf.splice_left), dtype=feats.dtype)
    feats_right = np.empty(shape=(M,N*conf.splice_right), dtype=feats.dtype)
    feats_padded = np.pad(feats,((conf.splice_left, conf.splice_right),(0,0)),mode=conf.splice_mode)
    for m in range(M):
        (i,j,k) = (m, m+conf.splice_left, m+conf.splice_left+conf.splice_right)
        feats_left[i] = feats_padded[i:j].flatten()
        feats_right[i] = feats_padded[j+1:k+1].flatten()

    if conf.splice_mode == 'empty':
        (i,j) = (conf.splice_left, M-conf.splice_right)
        return np.concatenate((feats_left[i:j],feats[i:j],feats_right[i:j]),axis=1)
    else:
        return np.concatenate((feats_left,feats,feats_right),axis=1)
