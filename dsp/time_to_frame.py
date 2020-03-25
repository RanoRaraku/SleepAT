"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic DPS library.
"""
import numpy as np
#------------------------------------------------


def time_to_frame(t:float, wstep:float) -> np.ndarray:
    """
    Convert time instant 't' to a frame index of a signal segmented using dsp.segment.
    The time instant marks onset from signal beginning, i.e. t = 0 is the beginning.
    Used to transform event boundaries for target encoding. We make no check about sanity
    of produced indeces (are they in/out of bounds of a signal).
    Input:
        t .... a time instant in seconds where t = 0 means beginning.
        wstep .... window step in seconds (default: float = 0.01) 
    """
    return np.int(np.around(t/wstep))
