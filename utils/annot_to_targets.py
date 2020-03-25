"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
import math
import numpy as np
from sleepat.dsp import time_to_frame
from sleepat.base.opts import AnnotToTargetsOpts

def annot_to_targets(annot:dict, dur:float, classes:dict,
    config:str=None, **kwargs) -> None:
    """
    Transform annotation for an segment of duration 'dur' to targets
    for training. Uses the same segmentation setup as dsp.segment().
    Events in annotation must be defined in classes and are expected
    to be ordinal numbers.
    Input:
        annot ... data source directory
        dur .... duration in seconds
        classes .... maps event labels to ordinal numbers
        <wlen> ...
        <wstep> ....
        config ....
        **kwargs .... allow to set wstep from a config file
    """
    conf = AnnotToTargetsOpts(config,**kwargs)
    fnum = math.floor((dur - conf.wlen)/conf.wstep) + 1    
    target = np.zeros(shape=(fnum,),dtype = np.float32) + classes['null']

    if annot is None:
        print('No events in annotation convert, returning /null/ labels.')
        return target
    for event in annot:
        ii =  time_to_frame(event['onset'],conf.wstep)
        jj = ii + time_to_frame(event['duration'],conf.wstep)
        target[ii:jj] = classes[event['label']]
    return target
