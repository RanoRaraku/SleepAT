"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
import datetime
from datetime import timedelta
import sleepat
from sleepat import utils

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
        seg_start = utils.string_to_date(periods['start']) + timedelta(seconds = seg['onset'])
        seg_period = seg
        seg_period['start'] = utils.date_to_string(seg_start)
        yield (seg_id, seg_period)
