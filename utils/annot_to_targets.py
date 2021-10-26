"""
Made by Michal Borsky, 2019, copyright (C) RU
Transforms textual annot to an array of numerical targets.
"""
import numpy as np
import sleepat
from sleepat import dsp, opts

def annot_to_targets(annot:list, duration:float, classes:dict, config:str=None, **kwargs) -> np.ndarray:
    """
    Transform annot from an segment of duration 'dur' to targets for training.
    Uses the same segmentation setup as dsp.segment(). classes in annot must be
    defined in classes and are expected to be integers.

    Arguments:
        annot ... a list of events contained in the recording
        duration ... recording duration in seconds
        classes ... maps event labels to ordinal numbers
        <wlen> ... window length used for segmentation (default:float=0.025)
        <wstep> ... window step used for segmentation (default:float=0.01)
        config .... configuration file for optional args <>.
        **kwargs .... to set optional args. <> from command line
    Return
        a numpy array of numerical targets 
    """
    conf = opts.AnnotToTargets(config,**kwargs)
    fnum = int((duration - conf.wlen) / conf.wstep ) + 1
    targets = np.zeros(shape=(fnum,),dtype = np.float32) + classes['null']

    if annot is None:
        print('No classes in annot to convert, returning /null/ labels.')
        return targets
    for item in annot:
        ii =  dsp.time_to_frame(item['onset'],conf.wstep)
        jj = ii + dsp.time_to_frame(item['duration'],conf.wstep)
        targets[ii:jj] = classes[item['label']]
    return targets
