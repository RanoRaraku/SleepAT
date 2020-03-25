"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
import copy
import numpy as np

def segment_events(events:list, segments:dict) -> dict:
    """
    Segment list of events, where each event is a dictionary, according to
    the segmentation dict, events can be split into multiple segments.
    Assumptions about segments and events:
    1) Events and segments are chornologically ordered by their onsets,
    2) They use the same units (i.e. seconds or samples).
    3) There is no overlap between events.

    The function is a generator, the accumulation happens on the consumer part.
    Input:
        events .... a list of events, each event is a dictionary (default:list).
        segments .... a dict of segments, each segment is a dictionary (default:dict).
    Output:
        a tuple of (seg_id, seg_events).
    """
    if segments is None or len(segments) == 0:
        return ('0000', events)

    lower = 0
    for seg_id, seg in segments.items():
        seg_beg = seg['onset']
        seg_end = np.round(seg['onset']+seg['duration'],5)
        out = list()

        for idx, event in enumerate(events[lower:]):
            event_beg = event['onset']
            event_end = np.round(event['onset']+event['duration'], 5)

            # 0) Event is cleanly before the segment.
            if event_end <= seg_beg:
                continue
            # 1) Event starts before, but ends within/after segment.
            if event_beg < seg_beg:
                event_beg = seg_beg
                event['onset'] = event_beg
                event['duration'] = np.round(event_end-event_beg,5)
            # 2,3,4) Event ends within segment.
            if event_end <= seg_end:
                event['onset'] = np.round(event_beg-seg_beg, 5)
                out.append(event)
                continue
            # 5) Event starts within/before, but ends after segment.
            if event_beg < seg_end:
                tmp = event.copy()
                tmp['onset'] = np.round(event_beg-seg_beg, 5)
                tmp['duration'] = np.round(seg_end-event_beg, 5)
                out.append(tmp)
            # 6) Event is cleanly after segment.
            # Note: Shared with previous case due to Assumption 3)
            lower += idx
            break
        yield (seg_id, out)
