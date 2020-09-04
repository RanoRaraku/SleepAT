"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic DPS library.
"""

def time_to_frame(t:float, wstep:float) -> int:
    """
    Convert time instant 't' to a frame index of a signal segmented
    using dsp.segment(). The time instant marks onset from signal
    beginning, i.e. t = 0 is the beginning. Used to transform event
    boundaries for target encoding, i.e. Makes no check about sanity
    of produced indeces (are they in/out of bounds of a signal).
    Input:
        t ... time instant in seconds
        wstep ... window step in seconds
    Output:
        frame index
    """
    return int(t/wstep)
