"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic DPS library.
"""
import numpy as np

def bark(f:float) -> np.ndarray:
    """
    Conversion from linear frequency to bark-freqency scale
    ------------------------------------------------------
    Input : f .... frequency in Hz, (default: float=0.0)
    Output: barkf .... frequency in bark
    """
    return 6*np.log(f/600 + np.sqrt((f/600)**2 + 1))