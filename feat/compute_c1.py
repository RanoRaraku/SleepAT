"""
Made by Michal Borsky, 2019, copyright (C) RU
Acoustuc feature extraction library.
"""
import numpy as np
import sleepat
from sleepat import dsp, opts

def compute_c1(sig, config:str=None, **kwargs) -> np.ndarray:
    """
    Calculates normalized auto-correlation coefficient at unit sample delay.
    Uses slightly different formula from [Atal76]
    ------------------------------------------------------------------------
    Input :
        sig ... a numpy array of values
        <fs> .... sampling frequency in Hz (default:float = 8000
        <wlen> ... window length in seconds (default:float = 0.25)
        <wstep> ... window step in seconds (default:float = 0.01)
        config ...
        **kwargs ...
    Output :
        c1 ... numpy.ndarray(shape=(num_frames,1), dtype=numpy.float)
    """
    conf = opts.C1Opts(config,**kwargs)

    frames = dsp.segment(sig, conf.fs, conf.wlen, conf.wstep, conf.wtype)
    c1 = np.zeros(shape=(frames.shape[0],1))
    for index, frame in enumerate(frames, start=0):
        num = np.dot(frame[:-1], frame[1:])
        den = np.sqrt(np.dot(frame[1:],frame[1:])*np.dot(frame[:-1],frame[:-1]))
        c1[index]  = num/den
    return c1.astype(np.float32)
