"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
from .date_to_string import date_to_string
from .string_to_date import string_to_date
from datetime import timedelta

def segment_stamps(stamp, segments:dict=None) -> tuple:
    """
    Segment a waveform with the sampling frequency fs according to a dict of segments.
    We assume a segment is a dict of {'onset' = float, 'duration' = float, 'id' = string}.
    We assume the onset and duration are in seconds, not samples.

    The function is a generator, the accumulation happens on the consumer part.
    Input:
        stamps ....
        fs .... sampling frequency in Hz, (default:float = 8000).
        segments .... a dictionary of segments, see create.segments(), (default:dict = None).
    Output:
        a tuple of (seg_id, seg_wave, seg_duration).
    """
    if segments is None or len(segments) == 0:
        return ('0000', stamp)

    for seg_id, item in segments.items():
        seg_start = string_to_date(stamp['start']) + timedelta(seconds = item['onset'])
        seg_end = seg_start + timedelta(seconds=item['duration'])
        yield (seg_id, {'start':date_to_string(seg_start),'end':date_to_string(seg_end)})
