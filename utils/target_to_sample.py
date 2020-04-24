"""
    Made by Michal Borsky, 2019
    Copyright (C) RU
    Library to handle target encoding and manipulation for ML training.
"""
import numpy as np
import math

def target_to_sample(target:np.ndarray, fs:float=8000, wstep:float=0.01, **kwargs):
    """
    Converts targets at frame level to targets at sample level.
    Used for visualization where we draw boundaries around events
    in the time domain.
    """
    fstep = math.ceil(wstep*fs)
    return(np.repeat(target,fstep))
