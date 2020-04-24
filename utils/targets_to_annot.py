"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
import datetime
from datetime import timedelta
import numpy as np
import sleepat
from sleepat import dsp, opts, utils


def targets_to_annot(targets:dict, classes:dict, config:str=None, **kwargs) -> None:
    """
    Transform targets obtained during inference to annotation.
    Uses the same segmentation setup as dsp.segment(). Targets
    is an array of integers, labels are defined in classes. No
    processing of targets is done here, anything of length > 0
    is an event. All processing must be done before. Onset is
    calculated with respect to the beginning of the array, but
    utt timestamp can be supplied to position the event correctly.
    If not supplied field is left empty. Only non-null targets
    are transformed to events by default. Null targets are usually
    silence. Optional arguments <> can be set from config or as kwargs.

    Arguments:
        annot ... data source directory
        classes .... dictionary to map event labels to ordinal numbers
        <wlen> ... window length used for segmentation (default:float=0.025)
        <wstep> .... window step used for segmentation (default:float=0.01)
        <utt_timestamp> ... reference timestamp to anchor event start (default:str = '')
        <no_null> ... dont transform 'null' targets to events (default:bool = True)
        config .... configuration file for optional args. <>
        **kwargs .... to set optional args. <> from command line
    Return:
        annot ... a list of dicts containing events
    """
    conf = opts.TargetsToAnnotOpts(config,**kwargs)
    const = int(conf.wlen/conf.wstep) - 1   # To compensate for segmentation overlap

    if isinstance(targets,list):
        targets = np.array(targets)
    if targets.size == 0:
        print(f'Error: targets array is empty.')
        exit(1)

    tgt2lab = dict()
    if not classes:
        print(f'Error: classes is empty, use utils.make_ali() to make just alignment.')
        exit(1)
    for key, item in classes.items():
        if item in tgt2lab and key != 'null':
            continue
        tgt2lab[item] = key

    annot = list()
    tgt_diff = (np.nonzero(np.diff(targets))[0]+1).tolist()
    event_beg, event_end = tgt_diff.copy(), tgt_diff.copy()
    event_beg.insert(0,0)
    event_end.append(targets.size+const)
    for beg,end in zip(event_beg,event_end):
        label = tgt2lab[targets[beg]]
        if label != 'null' or not conf.no_null:
            onset = dsp.frame_to_time(beg,conf.wstep)
            dur = dsp.frame_to_time(end-beg,conf.wstep)
            start = utils.date_to_string(
                utils.string_to_date(conf.utt_timestamp)+timedelta(seconds=onset))
            annot += [{'label':label, 'start':start, 'onset':onset, 'duration':dur}]
    return annot