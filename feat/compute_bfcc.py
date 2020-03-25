"""
Made by Michal Borsky, 2019, copyright (C) RU
Acoustuc feature extraction library.
"""
import numpy as np
from scipy.fftpack import dct
from sleepat.base.opts import BfccOpts
from sleepat.dsp import barkfb, preemphasis, pow_spect

def compute_bfcc(sig, config:str=None, **kwargs) -> np.ndarray:
    """
    Calculates bark-frequency cepstral coefficients. The following parameters
    can be set manually from config file or as kwargs. Kwargs take precence
    for conflicting values. Check opts.BfccOpts() for default values.
    -------------------------------------------------------------------------
    Input :
        sig ... a numpy array of values
        <fs> .... sampling frequency in Hz (default: float = 8000)
        <preemphasis_alpha> ... pre-emphasis coefficient (default: float = 0.97)
        <wlen> ... window length in seconds (default: float = 0.25)
        <wstep> ... window step in seconds (default: float = 0.01)cd ..
        <fmin> ... minimal frequency (default: float = 0)
        <fmax> ... maximum frequency (default: float = fs/2)
        <nceps> ... number of coefficients including 0th (default: int = 13)
        <use_log_fbank> .... use log on fbank energies (default:bool = True)
        config ...
        **kwargs ...
    Output :
        numpy.ndarray(shape=(num_frames,nceps), dtype=numpy.float)
    """
    conf = BfccOpts(config=config, **kwargs)

    conf.nfft = np.around(conf.wlen*conf.fs).astype(np.uint16)
    sig = preemphasis(sig, conf.preemphasis_alpha)
    pow_frames = pow_spect(sig, conf.fs, conf.wlen, conf.wstep, conf.remove_dc, conf.wtype)
    fbank = barkfb(conf.fs, conf.nfft, conf.fmin, conf.fmax)
    fbank_feats = np.dot(pow_frames,fbank.T)
    if conf.use_log_fbank:
        #fbank_feats = np.where(fbank_feats == 0, np.finfo(float).eps, fbank_feats)  # Numerical Stability
        fbank_feats = 10*np.log10(fbank_feats + np.finfo(float).eps)
    bfcc = dct(fbank_feats, type=2, axis=1, norm='ortho')
    return bfcc[:,:conf.nceps].astype(np.float32)
