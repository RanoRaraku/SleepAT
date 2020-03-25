"""
Made by Michal Borsky, 2019, copyright (C) RU
Get rid of math - it does interger rounding, replace with np routine
"""
import numpy as np
import math
#------------------------------------------------

def segment(sig, fs:float, wlen:float, wstep:float, remove_dc:bool,
    wtype:str) -> np.ndarray:
    """
    Segments the signal into frames and applies a window function.
    Input:
        sig  .... a (t,1) numpy.ndarray
        fs .... sampling frequency (default:float = 8000)
        wlen ... window length in seconds (default:float = 0.025)
        wstep ... window step in seconds (default:float = 0.01)
        wtype ... window type ("hamming"|"bartlett"|"blackman"|"hanning"|"kaiser|rectangular")
                (default:str = hamming)
        remove_dc ... removes offset, done on per-segment basis (default:bool = True)
        kwargs ... optional arguments. Used to set beta for Kaiser window
    Output:
        frames ... numpy.ndarray(shape=(num_of_frames,flen), dtype=numpy.float32)
    """
    flen = math.ceil(wlen*fs)
    fstep = math.ceil(wstep*fs)
    fnum = math.floor((len(sig)-flen)/fstep) + 1
    frames = np.zeros(shape=(fnum,flen),dtype = np.float32)

    if wtype == 'rectangular':
        win = np.ones(shape=(flen,1))
    else:
        window = getattr(np,wtype)
        win = window(flen)

    # Main loop
    for i in np.arange(fnum):
        ii = i*fstep
        jj = ii + flen
        frame = sig[ii:jj]*win
        if remove_dc:
            frame -= np.mean(frame)
        frames[i] = frame.astype(np.float32)
    return frames
