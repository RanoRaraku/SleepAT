"""
Made by Michal Borsky, 2020, under Apache License v2.0 license.
Evaluate identification/detection error.
"""
import numpy as np

def eval_ser(ref:list, hyp:list, events:dict) -> np.ndarray:
    """
    Evaluate segmentation error at a continuous levels. Inputs are 'ref', 'hyp',
    which are lists, and 'events' which is a dict. Each event in hyp/ref  has 'onset',
    duration', and 'label' keys defined, i.e. {'label':snore, 'onset':0.0, 'duration': 1.0}.
    'Events' dict maps between event labels (string) and event symbols (int) i.e. {'snore':1}. 
    Used in case multiple labels (central/obstructite apnea) map to the same symbol. All
    events in ref and hyp are included in evaluation, to remove those that are not to be scored
    beforehand. Ref/events can't be empty, but they can be of unequal length. The implementation
    assumes events are chronologically ordered and don't overlap.

    The protocol is to calculate overlap between two events in seconds, construct overlap matrix
    O and identity matrix I. It is possible for one event to contribute to many. We calculate the
    contribution towards 4 measures in seconds as follows:
    Hit        : overlap for events with same labels
    Confusion  : overlap for events with different labels
    Miss       : complement to ref. events
    False Alarm: complement to hyp. events

    Arguments:
        ref ... a list of reference events, each event is a dictionary
        hyp ... a list of hypothesis events, each event is a dictionary
        events ...  maps event labels (str) to symbols (int)
    Return:
        score as numpy array (h/m/fa/c)
    """
    # Checks
    for item in [ref,events]:
        if not item:
            print(f'Error eval_ser(): {item} is empty.')
            exit(1)
    for item in [ref,hyp]:
        if not isinstance(item,list):
            print(f'Error eval_ser(): {item} is not a list.')
            exit(1)
    if not isinstance(events,dict):
        print(f'Error eval_ser(): events is not a dict.')
        exit(1)
    if not hyp:
        refdur = 0.0
        for re in ref:
            refdur += re['duration']
        return np.array([0,refdur,0,0],dtype=np.float32)


    # Compute Overlap and Identity matrices, negative overlaps are set to 0
    (refdur, hypdur) = (0.0,0.0)
    (reflen, hyplen) = len(ref),len(hyp)
    O = np.zeros(shape=(reflen,hyplen),dtype=np.float32)
    I = np.zeros(shape=(reflen,hyplen),dtype=np.bool)

    for i,re in enumerate(ref):
        for j,he in enumerate(hyp):
            (re_on,re_dur) = re['onset'],re['duration']
            (he_on,he_dur) = he['onset'],he['duration']
            O[i,j] = max(0, min(re_on+re_dur,he_on+he_dur) - max(re_on,he_on))
            I[i,j] = (events[re['label']] == events[he['label']])

            # Accumulate total ref and hyp durations
            hypdur += (he_dur*(i==0))
        refdur += re_dur


    # Assign labels
    h = np.multiply(O,I).sum()
    c = O.sum() - h
    m = round(refdur - h - c, 8)
    fa = round(hypdur - h - c, 8)
    score = np.array([h,m,fa,c],dtype=np.float32)

    return score
