"""
Made by Michal Borsky, 2019, copyright (C) RU
Acoustuc feature extraction library.
"""
import numpy as np
from sleepat.base.opts import SpectSlopeOpts
from sleepat.dsp import pow_spect

def compute_spect_slope(sig, config:str=None, **kwargs) -> np.ndarray:
    """
    Calculates the exponential spectral decay in freq. bands.
    ----------------------------------------------------------
    Input :
        sig ... a numpy array of values
        <fs>  ... sampling frequency in Hz (def:float = 8000)
        <ss_bands> ...a array(shape=(N,2)) that contains ([band_start, band_stop])
                    in Hz on each row (def:float = [0,fs/2])
        <wlen> ... window length in ms (def:float = 0.25)
        <wstep> ... window step in ms (def:float = 0.01)
        config ...
        **kwargs ...
    Output :
        spect_slope ... numpy.ndarray(shape=(num_frames,num_bands))
    """
    conf = SpectSlopeOpts(config,**kwargs)

    bins = np.round(conf.ss_bands*conf.wlen)
    pow_frames = pow_spect(sig, conf.fs, conf.wlen, conf.wstep, conf.remove_dc, conf.wtype)
    log_frames = 10*np.log10(pow_frames + np.finfo(float).eps)
    spect_slope = np.zeros(shape=(pow_frames.shape[0],bins.shape[0]))
    for i in range(log_frames.shape[0]):
        for j in range(bins.shape[0]):
            y = log_frames[i,int(bins[j,0]):int(bins[j,1])]
            x = range(y.size)
            spect_slope[i,j] = np.polyfit(x, y, deg = 1)[0]
    return spect_slope.astype(np.float32)
