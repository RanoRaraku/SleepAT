"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
import math
import numpy as np
import sleepat
from sleepat import dsp, opts

def annot_to_targets(annot:dict, dur:float, classes:dict,
    config:str=None, **kwargs) -> None:
    """
    Transform annotation for an segment of duration 'dur' to targets
    for training. Uses the same segmentation setup as dsp.segment().
    Events in annotation must be defined in classes and are expected
    to be integers. 

    Arguments:
        annot ... data source directory
        dur .... duration in seconds
        classes .... maps event labels to ordinal numbers
        <wlen> ... window length used for segmentation (default:float=0.025)
        <wstep> .... window step used for segmentation (default:float=0.01)
        config .... configuration file for optional args <>.
        **kwargs .... to set optional args. <> from command line
    """
    conf = opts.AnnotToTargetsOpts(config,**kwargs)
    fnum = math.floor((dur - conf.wlen)/conf.wstep) + 1    
    target = np.zeros(shape=(fnum,),dtype = np.float32) + classes['null']

    if annot is None:
        print('No events in annotation convert, returning /null/ labels.')
        return target
    for event in annot:
        ii =  dsp.time_to_frame(event['onset'],conf.wstep)
        jj = ii + dsp.time_to_frame(event['duration'],conf.wstep)
        target[ii:jj] = classes[event['label']]
    return target
