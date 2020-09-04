"""
Made by Michal Borsky, 2019, copyright (C) RU
Transforms textual scoring to an array of numerical targets.
"""
import numpy as np
import sleepat
from sleepat import dsp, opts

def scoring_to_targets(scoring:list, period:dict, events:dict, config:str=None, **kwargs) -> np.ndarray:
    """
    Transform scoring from an segment of duration 'dur' to targets for training.
    Uses the same segmentation setup as dsp.segment(). Events in scoring must be
    defined in events and are expected to be integers.

    Arguments:
        scoring ... a list of with events for one utterance/segment
        period ... utterance information containing 'duration' field
        events ... maps event labels to ordinal numbers
        <wlen> ... window length used for segmentation (default:float=0.025)
        <wstep> ... window step used for segmentation (default:float=0.01)
        config .... configuration file for optional args <>.
        **kwargs .... to set optional args. <> from command line
    Return
        a numpy array of numerical targets 
    """
    conf = opts.AnnotToTargets(config,**kwargs)
    fnum = int((period['duration'] - conf.wlen) / conf.wstep ) + 1
    targets = np.zeros(shape=(fnum,),dtype = np.float32) + events['null']

    if scoring is None:
        print('No events in scoring to convert, returning /null/ labels.')
        return targets
    for item in scoring:
        ii =  dsp.time_to_frame(item['onset'],conf.wstep)
        jj = ii + dsp.time_to_frame(item['duration'],conf.wstep)
        targets[ii:jj] = events[item['label']]
    return targets
