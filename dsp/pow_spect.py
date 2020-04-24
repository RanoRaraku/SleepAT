"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic DPS library.
"""
import numpy as np
import sleepat
from sleepat import dsp

def pow_spect(x, fs:float, wlen:float, wstep:float, remove_dc:bool,
    wtype:str) -> np.ndarray:
    """
    Computes power spectrum of a signal with segmentation.
    Input:
        x  .... input signal as a np.ndarray
        fs .... sampling frequency in Hz 
        wlen .... window length in seconds 
        wstep .... window step in seconds
    Output:
        power spectrum as np.ndarray(shape=(num_of_frames,wlen),dtype=np.float32)
        that contains power spectrum
    """
    X = dsp.spect(x,fs,wlen,wstep,remove_dc,wtype)
    N = 1/(X.shape[0])
    return N*(X**2).astype(np.float32)
