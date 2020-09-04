"""
Made by Michal Borsky, 2019, copyright (C) RU
Get rid of math - it does interger rounding, replace with np routine
"""
import numpy as np
import math

def segment(x, fs:float, wlen:float, wstep:float, remove_dc:bool, wtype:str) -> np.ndarray:
    """
    Segments the signal into frames and applies a window function.
    Input:
        x  .... input signal as (t,1) numpy.ndarray
        fs .... sampling frequency (default:float = 8000)
        wlen ... window length in seconds (default:float = 0.025)
        wstep ... window step in seconds (default:float = 0.01)
        wtype ... window type ("hamming"|"bartlett"|"blackman"|"hanning"|"rectangular")
            (default:str = hamming)
        remove_dc ... removes offset, done on per-segment basis (default:bool = True)
    Output:
        frames ... signal frames oragnized as a matrix of shape=(num_of_frames,flen)
    """
    flen = math.ceil(wlen*fs)
    fstep = math.ceil(wstep*fs)
    fnum = math.floor((len(x)-flen)/fstep) + 1
    frames = np.zeros(shape=(fnum,flen),dtype=np.float32)

    if wtype == 'rectangular':
        win = np.ones(shape=(flen,1))
    else:
        window = getattr(np,wtype)
        win = window(flen)

    # Main loop
    for i in range(fnum):
        ii = i*fstep
        jj = ii + flen
        frame = x[ii:jj]*win
        if remove_dc:
            frame -= np.mean(frame)
        frames[i] = frame.astype(np.float32)
    return frames
