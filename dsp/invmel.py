"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic DPS library.
"""

def invmel(melf:float) -> float:
    """
    Conversion from mel-freqency to linear frequency scale.
    ------------------------------------------------------
    Input : melf .... frequency in mel
    Output: f ...frequency in Hz
    """
    return 700*(10**(melf/2595)-1)