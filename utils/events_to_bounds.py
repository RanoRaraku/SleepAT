"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
import numpy as np

def events_to_bounds(events:list=None) -> list:
    """
    Transforms any list of dictionaries into a numpy array that contains start and end of all events.
    Each dictionary has to have these two keys defined. Works for annotation, segments. The ends are
    calculated as onset + duration.
    Input:
        events .... a list of events where each event is a dictionary (default:list = None).
    Output
        out .... a numpy array in ([start,end]) format
    """
    out = np.zeros(shape=(len(events),2))
    for i,event in enumerate(events):
        out[i,0] = event['onset']
        out[i,1] = event['onset'] + event['duration']
    return out
