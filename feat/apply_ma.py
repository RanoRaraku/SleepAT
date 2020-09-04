"""
Made by Michal Borsky, 2019, copyright (C) RU
Acoustuc feature extraction library.
"""
import numpy as np
import sleepat
from sleepat import opts

def apply_ma(feats:np.ndarray, config:str=None, **kwargs) -> np.ndarray:
    """
    Apply moving average filter of 'ma_weights' across feature domain. The mask is
    normalized to sum to 1 to avoid scaling features. The idea is to smooth
    feature contours, prior to any processing. Can be applied to any numpy arrays
    (i.e. posteriors). Supports two-way moving average to offset introducing shift.

    Input :
        feats ... a feature vector of np.ndarray(shape=(N,M)) type.
        <ma_weights> ... a mask of floats (default:list = [.33,.33.,.33])
        <ma_mode> ... 'one-way'|'two-way' movavg. pass (default:str = 'two-way')
    Output:
        np.ndarray(shape=(N, M), dtype=feats.dtype)
    """
    conf = opts.ApplyMa(config,**kwargs)

    if isinstance(conf.ma_weights,int):
        n = conf.ma_weights
        conf.ma_weights = np.ones(shape=(n,),dtype=np.float32) / n
    elif isinstance(conf.ma_weights,list):
        n = len(conf.ma_weights)
        conf.ma_weights = np.ndarray(conf.ma_weights,dtype=np.float32) / sum(conf.ma_weights)
    else:
        print(f'Error: ma_weights needs to be either int of an 1D array.')
        exit(1)

    # Forwards backward MA to offset shift
    feats = np.pad(feats, ((n-1,0),(0,0)), mode='edge')
    if conf.ma_mode == 'two-way':
        feats = np.pad(feats, ((0,n-1),(0,0)), mode='edge')

    feats.cumsum(axis=0,out=feats)
    feats[n:,:] -= feats[:-n,:]
    feats = feats[n-1:,:] / n
    if conf.ma_mode == 'two-way':
        feats = feats[::-1]
        feats.cumsum(axis=0,out=feats)
        feats[n:,:] -= feats[:-n,:]
        feats = feats[::-1]
        feats = feats[:-n+1,:] / n

    return feats
