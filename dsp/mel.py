"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic DPS library.
"""
import numpy as np

def mel(f:float) -> np.ndarray:
    """
    Conversion from linear frequency to mel-freqency scale
    ------------------------------------------------------
    Input : f .... frequency in Hz, (default: float=0.0)
    Output: melf .... frequency in mel
    """
    return 2595*np.log10(1+f/700)
