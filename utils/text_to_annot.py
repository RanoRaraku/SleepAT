"""
Made by Michal Borsky, 2019, copyright (C) RU
Transforms textual scoring to an array of numerical targets.
"""
import re
import numpy as np

def text_to_annot(text:str, duration:float, lexicon:dict) -> list:
    """
    Transform annot from an segment of duration 'dur' to targets for training.
    Uses the same segmentation setup as dsp.segment(). classes in scoring must be
    defined in classes and are expected to be integers.

    Arguments:
        text ... a list of with classes for one recording
        duration ... recording duration in seconds
        lexicon ... maps words to sequence of phones
        config .... configuration file for optional args <>.
        **kwargs .... to set optional args. <> from command line
    Return
        a numpy array of numerical targets 
    """

    #conf.add_eps = True
    add_eps = True

    if not isinstance(text,str):
        print(f'Error: text expected string, got {type(text)}')
        exit(1)
    if not isinstance(duration,float):
        print(f'Error: duration expected float, got {type(text)}')
        exit(1)
    if not isinstance(text,'str'):
        print(f'Error: lexicon expected dict, got {type(text)}')
        exit(1)                


    if add_eps == True:
        lexicon_eps = dict()
        for key, val in lexicon.items():
            re.findall(r'((\S)\2{2,})', val)
            val = val
            lexicon_eps[key] = val


    # Convert text to subword units based on lexicon 
    text_phn = 'sil '
    for word in text.split():
        text_phn += f'{lexicon[word]} eps '
    text_phn += 'sil'

    # Add eps between each repeating subword unit

    



    fnum = int((duration - conf.wlen) / conf.wstep ) + 1
    targets = np.zeros(shape=(fnum,),dtype = np.float32) + classes['null']

    if text is None:
        print('No classes in scoring to convert, returning /null/ labels.')
        return targets
    for word in text:
        ii =  dsp.time_to_frame(item['onset'],conf.wstep)
        jj = ii + dsp.time_to_frame(item['duration'],conf.wstep)
        targets[ii:jj] = classes[item['label']]
    return targets
