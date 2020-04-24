"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic DPS library.
"""
import numpy as np

def preemphasis(x, preemphasis_alpha:float) -> np.ndarray:
    """
    Applies the preemphasis to a signal using the formula
    Input:
        x .... singal as (t,1) numpy.ndarray
        alpha .... pre-emphasis coefficient (default:float = 0.97)
    Output:
        a numpy array of the same dimensions
    """
    if preemphasis_alpha < 0:
        raise ValueError('Filter coefficient alpha must be a float in <0,1> range')
    return np.append(x[0],x[1:]-preemphasis_alpha*x[:-1])
