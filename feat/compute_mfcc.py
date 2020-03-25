"""
Made by Michal Borsky, 2019, copyright (C) RU
Acoustuc feature extraction library.
"""
import numpy as np
from scipy.fftpack import dct
from sleepat.base.opts import MfccOpts
from sleepat.dsp import melfb, preemphasis, pow_spect

def compute_mfcc(sig, config:str=None, **kwargs) -> np.ndarray:
    """
    Calculates mel-frequency cepstral coefficients. The following parameters
    can be set manually, from config file or as kwargs. Kwargs take precence
    for conflicting values. Check opts.BfccOpts() for default values
    -----------------------------------------------------------------------
    Input :
        sig ... a numpy array of values
        <fs> .... sampling frequency in Hz (default: float = 8000)
        <preemphasis_alpha> ... pre-emphasis coefficient (default: float = 0.97)
        <wlen> ... window length in seconds (default: float = 0.25)
        <wstep> ... window step in seconds (default: float = 0.01)
        <mel_filts>  .... number of filters (default: int = 22)
        <fmin> ... minimal frequency (default: float = 0)
        <fmax> ... maximum frequency (default: float = fs/2)
        <nceps> ... num. of cepstral coefficients including 0th (default:int = 13)
        config ...
        **kwargs ...
    Output :
        numpy.ndarray(shape=(num_frames,nceps), dtype=numpy.float)
    """
    conf = MfccOpts(config,**kwargs)

    nfft = np.around(conf.wlen*conf.fs).astype(np.uint16)
    melfb = melfb(conf.mel_filts, conf.fs, nfft, conf.fmin, conf.fmax)
    sig = preemphasis(sig, conf.preemphasis_alpha)
    pow_frames = pow_spect(sig, conf.fs, conf.wlen, conf.wstep, conf.remove_dc, conf.wtype)
    fbanks = np.dot(pow_frames,melfb.T)
    if conf.use_log_fbank:
        fbanks = np.where(fbanks == 0, np.finfo(float).eps, fbanks)  # Numerical Stability
        fbanks = 10*np.log10(fbanks+np.finfo(float).eps)
    mfcc = dct(fbanks, type=2, axis=1, norm='ortho')
    return mfcc[:,:conf.nceps].astype(np.float32)
