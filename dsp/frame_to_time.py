"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic DPS library.
"""

def frame_to_time(idx:int, wstep:float) -> float:
    """
    Convert frame index 'idx' to a time instant using same approach as
    in dps.segment(). The instant corresponds to the beginning
    of the frame. We make no check about sanity of produced instants
    (are they in/out of bounds of a signal).
    Input:
        idx ... frame index as integer.
        wstep ... window step in seconds (default: float = 0.01)
    """
    return round(idx*wstep,5)