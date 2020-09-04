"""
Made by Michal Borsky, 2019, copyright (C) RU
Transform an array of numerical targets to textual scoring.
"""
import datetime
from datetime import timedelta
import numpy as np
import sleepat
from sleepat import dsp, opts, utils


def targets_to_scoring(targets:np.ndarray, post:np.ndarray, events:dict, config:str=None,**kwargs) -> list:
    """
    Transform targets obtained during inference to scoring. Uses the same segmentation
    setup as dsp.segment(). Targets is an array of integers, labels are defined in events.
    No processing of targets is done here, anything of length > 0 is an event. Processing
    is done before. Onset is calculated with respect to the beginning of the array, but
    utt timestamp can be supplied to align events correctly. If not supplied, the field
    is left empty. Only non-null targets are transformed to events by default.

    Arguments:
        targets ... a list/np.array of tartgets as numbers
        events ... dictionary to map event labels to ordinal numbers
        <wlen> ... window length used for segmentation (default:float=0.025)
        <wstep> .... window step used for segmentation (default:float=0.01)
        <timestamp> ... reference timestamp to anchor event start (default:str = '')
        config .... configuration file for optional args. <>
        **kwargs .... to set optional args. <> from command line
    Return:
        scoring ... a list of dicts containing events
    """
    conf = opts.TargetsToAnnot(config,**kwargs)
    delta = int(conf.wlen/conf.wstep) - 1   # To compensate for segmentation overlap
    events_inv = dict()
    scoring = list()

    if isinstance(targets,list):
        targets = np.array(targets)
    if targets.size == 0:
        print(f'Error: targets array is empty.')
        exit(1)
    if not events:
        print(f'Error: events is empty, use utils.make_ali() to make just alignment.')
        exit(1)
    for key, val in events.items():
        if val in events_inv and key != 'null':
            continue
        events_inv[val] = key

    tgt_diff = (np.nonzero(np.diff(targets))[0] + 1).tolist()
    event_beg, event_end = tgt_diff.copy(), tgt_diff.copy()
    event_beg.insert(0,0)
    event_end.append(targets.size + delta)
    for beg,end in zip(event_beg, event_end):
        label = events_inv[targets[beg]]
        onset = round(dsp.frame_to_time(beg, conf.wstep),6)
        dur = round(dsp.frame_to_time(end - beg, conf.wstep),6)
        start = utils.date_to_string(utils.string_to_date(conf.tstamp) 
            + timedelta(seconds=onset))
        scoring += [{'label':label, 'start':start, 'onset':onset, 'duration':dur}]
    return scoring