"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic DPS library.
"""
import numpy as np
import sleepat
from sleepat import dsp

def melfb(mel_filts:int, fs:int, nfft:int, fmin:float, fmax:float) -> np.ndarray:
    """
    Creates mel-frequency filter bank. The filters are triangular
    in shape and have center freq. equdistantly placed on mel-freq.
    scale. The output is a matrix of weights, where each row is one
    filter.
    ----------------------------------------------------------
    Input :
        M  ... number of filters (default: int=23)
        fs ... sampling rate (default: int=16e3)
        nfft  ... DFT points (default: int=512)
        fmin ... low cutoff frequency for mel bank (default: float = 0.0)
        fmax ... high cutoff frequenct for mel bank (default: float = 8e3)
    Output :
        Wmel ... numpy.ndarray(shape=(M, nfft/2 + 1), dtype=float)
    """
    if fmax > fs/2:
        fmax = fs/2

    mel_points = np.linspace(dsp.mel(fmin), dsp.mel(fmax), mel_filts+2)
    hz_points = dsp.invmel(mel_points)
    bins = np.round(hz_points*nfft/fs).astype(np.uint32)
    Wmel = np.zeros(shape=(mel_filts, np.int(nfft/2 + 1)))

    for m in range(1, mel_filts+1):
        k_up = range(bins[m-1],bins[m])
        k_down = range(bins[m],bins[m+1])

        Wmel[m-1, k_up] = (k_up-bins[m-1])/(bins[m]-bins[m-1])
        Wmel[m-1, k_down] = (bins[m+1]-k_down)/(bins[m+1]-bins[m])
    return Wmel
