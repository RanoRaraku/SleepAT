"""
Made by Michal Borsky, 2020, copyright (C) RU.
Compute segmentation error rate between reference and hypothesis.
"""
import numpy as np

def compute_ser(ref:list, hyp:list, events:dict) -> dict:
    """
    Compute segmentation error at a continuous levels. Inputs are 'ref', 'hyp',
    which are lists of dicts, and 'events' which is a dict. Each event in hyp/ref  has 'onset',
    duration', and 'label' keys defined, i.e. {'label':snore, 'onset':0.0, 'duration': 1.0}.
    Events dict maps between event labels (string) event symbols (int) i.e. {'snore':1}. All
    event types in ref and hyp are included in evaluation, it might be prudent to remove some
    that are not to be scored before. Ref/hyp/events can't be empty, but they can be of unequal
    length. The implementation assumes events are chronologically ordered and don't overlap.

    The protocol is to calculate overlap between two events in seconds, construct overlap matrix
    O and identity matrix I. It is possible for one event to contribute to many. We calculate the
    contribution towards 4 measures in seconds as follows:
    Hit        : overlap for events with same labels
    Confusion  : overlap for events with different labels
    Miss       : complement parts of ref.events
    False Alarm: complement parts of hyp.events

    Arguments:
        ref ... a list of dicts indexed by utt_id with reference events
        hyp ... a list of dicts indexed by utt_id with hypothesis events
        events ...  maps event labels (str) to symbols (int)
    Return:
        score as numpy array (H/M/FA/C)
    """
    # Checks
    for item in [ref,events]:
        if not item:
            print(f'Error compute_ser(): {item} is empty.')
            exit(1)
    for item in [ref,hyp]:
        if not isinstance(item,list):
            print(f'Error compute_ser(): {item} is not a list.')
            exit(1)
    if not isinstance(events,dict):
        print(f'Error compute_ser(): events is not a dict.')
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


    # Compute error and Hit, Miss, False Alarm, Confusion
    h = np.multiply(O,I).sum()
    c = O.sum() - h
    m = round(refdur - h - c, 8)
    fa = round(hypdur - h - c, 8)
    score = np.array([h,m,fa,c],dtype=np.float32)

    return score