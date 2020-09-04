"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic DPS library.
"""
import numpy as np
import sleepat
from sleepat import dsp

def barkfb(fs:int, nfft:int, fmin:float, fmax:float) -> np.ndarray:
    """
    Calculates bark-frequency filter bank. The filters are trapeziodal
    in shape and have center freq. equdistantly placed on bark-freq.
    scale. The output is a matrix of weights, where each row is one
    filter.
    ----------------------------------------------
    Input :
        fs ... sampling rate (default: int=16e3)
        nfft  ... DFT points (default: int=512)
        fmin ... low cutoff frequency for mel bank (default: float=0.0)
        fmax ... high cutoff frequenct for mel bank (default: float=8e3)
    Output :
        Wbark ... numpy.ndarray(shape=(M, nfft/2 + 1), dtype=float)
    """
    if fmax > fs/2:
        fmax = fs/2

    bmin = dsp.bark(fmin)
    bmax = dsp.bark(fmax)
    bnum = int(bmax+0.5)
    bfreq = np.linspace(max(bmax/bnum, bmin), bmax, bnum)
    Wbark = np.zeros(shape=(bnum, int(nfft/2 + 1)))
    bark_bounds = np.array([-1.3,-0.5,0.5,2.5])

    # Equal loudness scaling as per "https://labrosa.ee.columbia.edu/matlab/rastamat/postaud.m"
    f2 = dsp.invbark(bfreq)**2
    eqnum = (f2/(f2 + 1.6e5))**2 * (f2 + 1.44e6)
    eqden = f2 + 9.61e6

    for j in range(bnum):
        hz_points = dsp.invbark(bark_bounds + bfreq[j])
        bins = np.round(hz_points*nfft/fs)
        bins = np.clip(bins, 0, int(nfft/2+1)).astype(np.uint32)
        k_up = np.arange(bins[0],bins[1])
        k_down = np.arange(bins[2]+1,bins[3])

        Wbark[j,k_up]  = 10**(2.5*(dsp.bark(k_up*fs/nfft) - bfreq[j]+0.5))
        Wbark[j,bins[1]:bins[2]+1] = 1
        Wbark[j,k_down] = 10**(0.5-dsp.bark(k_down*fs/nfft)+bfreq[j])
        Wbark[j,:] *= (eqnum[j]/eqden[j])
    return (Wbark**0.33)

    # Fixed sampling at 1 bark
    #filts = np.arange(max(1,math.ceil(bmin)), math.floor(bmax)+1)
