"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
import sleepat
from sleepat import utils, objects

def segment_periods(periods:dict, segments:dict=None) -> tuple:
    """
    Segment a waveform with the sampling frequency fs according to a dict of segments.
    We assume a segment is a dict of {'onset' = float, 'duration' = float, 'id' = string}.
    We assume the onset and duration are in seconds, not samples.

    The function is a generator, the accumulation happens on the consumer part.
    Input:
        stamps ....
        segments .... a dictionary of segments, see create.segments(), (default:dict = None).
    Output:
        a tuple of (seg_id, seg_wave, seg_duration).
    """
    if segments is None or len(segments) == 0:
        return ('0000', periods)

    for seg_id, seg in segments.items():
        seg_period = seg
        TStamp = objects.TimeStamp(tstamp=periods['start'])
        TStamp.increment(seg['onset'])
        seg_period['start'] = TStamp.as_string()
        yield (seg_id, seg_period)
