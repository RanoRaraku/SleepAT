"""
Made by Michal Borsky, 2019, copyright (C) RU
Acoustuc feature extraction library.
"""
import numpy as np
import sleepat
from sleepat import dsp, opts

def compute_fbank(sig, config:str=None, **kwargs) -> np.ndarray:
    """
    Calculates filter bank features. Works with Mel/Bark-filter banks.
    The following parameters can be set manually, from config file or
    as kwargs. Kwargs take precedence for conflicting values.
    Check opts.BfccOpts() for default values.
    ------------------------------------------------------------------
    Input :
        sig ... a numpy array of values
        <fbank_type> ... filter bank type (default:str = 'melfb')
        <fs> .... sampling frequency in Hz (default:float = 8000)
        <preemphasis_alpha> ... pre-emphasis coefficient (def:float = 0.97)
        <wlen> ... window length in seconds (def:float = 0.25)
        <wstep> ... window step in seconds (def:float = 0.01)
        <use_log> ... filter bank type mel/plp (def:bool = True)
        config ... an optional configuration file (def:str = None)
        **kwargs ...
    Output :
        fbank_feats ... numpy.ndarray(shape=(num_frames,M), dtype=numpy.float)
    """
    conf = opts.FbankOpts(config,**kwargs)

    # Main part
    nfft = np.around(conf.wlen*conf.fs).astype(np.uint16)
    if conf.fbank_type == 'melfb':
        fbank = melfb(conf.mel_filts, conf.fs, nfft, conf.fmin, conf.fmax)
    elif conf.fbank_type == 'barkfb':
        fbank = barkfb(conf.fs, nfft, conf.fmin, conf.fmax)

    sig = dsp.preemphasis(sig, conf.preemphasis_alpha)
    pow_frames = dsp.pow_spect(sig, conf.fs, conf.wlen, conf.wstep, conf.remove_dc, conf.wtype)
    fbank_feats = np.dot(pow_frames,fbank.T)
    if conf.use_log_fbank:
        fbank_feats = np.where(fbank_feats == 0, np.finfo(float).eps, fbank_feats)  # Numerical Stability
        fbank_feats = 10*np.log10(fbank_feats+np.finfo(float).eps)
    return fbank_feats.astype(np.float32)
