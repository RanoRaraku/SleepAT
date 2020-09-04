"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
import numpy as np

def segment_wave(wave:np.ndarray, fs:float=8000, segments:dict=None) -> np.ndarray:
    """
    Segment a waveform with the sampling frequency fs according to a dict of segments.
    We assume a segment is a dict of {'onset' = float, 'duration' = float, 'id' = string}.
    We assume the onset and duration are in seconds, not samples.

    The function is a generator, the accumulation happens on the consumer part.
    Input:
        wave .... a np.ndarray with the raw waveform.
        fs .... sampling frequency in Hz, (default:float = 8000).
        segments .... a dictionary of dictionaries, see create.segments(), (default:dict = None).
    Output:
        a tuple of (seg_id, seg_wave, seg_duration).
    """
    if segments is None or len(segments) == 0:
        return ('0000', wave)

    for seg_id, item in segments.items():
        seg_beg = int(item['onset']*fs)
        seg_end = int((item['onset']+item['duration'])*fs)
        if  seg_end > len(wave):
            seg_end = len(wave)
            print(f'{" ":3}Warning: segment {seg_id} is longer than waveform.')
        yield (seg_id, wave[seg_beg:seg_end])
