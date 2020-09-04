"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic DPS library.
"""
import numpy as np

def invbark(barkf:float) -> float:
    """
    Conversion from bark-freqency scale to linear frequency scale
    ------------------------------------------------------
    Input : barkf .... frequency in bark, (default: float=0.0)
    Output: f .... frequency in Hz
    """
    return 600*np.sinh(barkf/6)