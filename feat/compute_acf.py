"""
Made by Michal Borsky, 2019, copyright (C) RU
Acoustuc feature extraction library.
"""
import numpy as np
import sleepat
from sleepat import dsp, opts

def compute_acf(sig, config:str=None, **kwargs) -> np.ndarray:
    """
    Calculates autocorrelation features. The following parameters
    can be set manually, from config file or as kwargs. Kwargs take precence
    for conflicting values. Check opts.BfccOpts() for default values
    -----------------------------------------------------------------------
    Input :
        sig ... a numpy array of values
        <fs> .... sampling frequency in Hz (default: float = 8000)
        <preemphasis_alpha> ... pre-emphasis coefficient (default: float = 0.97)
        <wlen> ... window length in seconds (default: float = 0.25)
        <wstep> ... window step in seconds (default: float = 0.01)
        <acorr_type> auto-correlation type (default:str = )
        <nacf> ... num. of autocorr coefficients including (default:int = 320)
        config ...
        **kwargs ...
    Output :
        numpy.ndarray(shape=(num_frames,nceps), dtype=numpy.float)
    """
    conf = opts.Acf(config,**kwargs)

    sig = dsp.preemphasis(sig, conf.preemphasis_alpha)
    frames = dsp.segment(sig, conf.fs, conf.wlen, conf.wstep, conf.remove_dc, conf.wtype)
    M = frames.shape[0]  #pylint: disable=unsubscriptable-object
    acf = np.zeros(shape=(M,conf.nacf), dtype=np.float32)
    for i in range(M):
        acorr = np.correlate(frames[i,:], frames[i,:], mode=conf.actype)
        mid = int(acorr.shape[0]/2)
        acf[i] = acorr[mid:mid+conf.nacf]
    return acf.astype(np.float32)
