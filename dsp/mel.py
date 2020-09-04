"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic DPS library.
"""
import numpy as np

def mel(f:float) -> float:
    """
    Conversion from linear frequency to mel-freqency scale.
    ------------------------------------------------------
    Input : f .... frequency in Hz
    Output: melf .... frequency in mel
    """
    return 2595*np.log10(1+f/700)
