"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic DPS library.
"""
import numpy as np
from .segment import segment

def spect(sig, fs:float, wlen:float, wstep:float, remove_dc:bool,
    wtype:str) -> np.ndarray:
    """
    Computes magnitude spectrum of a signal with segmentation
    Input:
        sig  .... a (t,1) numpy.ndarray
        fs .... sampling frequency (default:float = 8000)
        wlen ... window length in seconds (default: float=0.025)
        wstep ... window step in seconds (default: float=0.01)
    Output:
       mag_frames ... a numpy.ndarray that contains magnitude spectra of frames
    """
    frames = segment(sig, fs, wlen, wstep, remove_dc, wtype)
    nfft = np.around(wlen*fs).astype(np.uint32)
    M = frames.shape[0]  #pylint: disable=unsubscriptable-object
    mag_frames = np.zeros(shape=(M, np.int(nfft/2 + 1)), dtype=np.float32)
    for i in np.arange(M):
        mag_frame = np.absolute(np.fft.rfft(frames[i, :], nfft))
        mag_frames[i, :] = mag_frame.astype(np.float32)
    return mag_frames