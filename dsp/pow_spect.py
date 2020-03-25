"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic DPS library.
"""
import numpy as np
from .spect import spect


def pow_spect(sig, fs:float, wlen:float, wstep:float, remove_dc:bool,
    wtype:str) -> np.ndarray:
    """
    Computes power spectrum of a signal with segmentation.
    Input:
        sig  .... a (t,1) numpy.ndarray
        fs .... sampling frequency
        wlen .... window length in seconds (default: float=0.025)
        wstep .... window step in seconds (default: float=0.01)
    Output:
        pow_spect .... a numpy.ndarray(shape=(num_of_frames,wlen),dtype=np.float32)
            that contains power spectrum
    """
    mag_spect = spect(sig,fs,wlen,wstep, remove_dc, wtype)
    return 1/mag_spect.shape[0]*(mag_spect**2).astype(np.float32) #pylint: disable=unsubscriptable-object