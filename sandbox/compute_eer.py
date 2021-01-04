"""
Made by Michal Borsky, 2019, copyright (C) RU
Compute identification error rate between reference and hypothesis.
"""
import numpy as np
import scipy
from scipy import ndimage

def compute_ier(ref:list, hyp:list, events:dict, thr:float = 0) -> dict:
    """
    Compute identification error rate (DER) at an event level. Inputs are 'ref', 'hyp', which are 
    lists, and 'events' which is a dict. Each event in hyp/ref  has 'onset', duration', and 'label'
    keys defined, i.e. {'label':snore, 'onset':0.0, 'duration': 1.0}. Events dict  maps between 
    event labels (string) event symbols (int) i.e. {'snore':1}. Threshold determines minimal value
    for successful detection, thr = 0 means any non-zero overlap is a hit. All event types in ref
    and hyp are included in evaluation, it might be prudent to remove some that are not
    to be scored before. Ref/hyp/events can't be empty, but they can be of unequal length.
    The implementation assumes events are chronologically ordered and don't overlap.

    We align ref/hyp using relative length of overlap (LoO),
    record (ref_i), (hyp_j) indexes, LoO value, and what event it was. The decision labels:
    Hit        :
    Confusion  :
    Miss       :
    False Alarm:

    The Event Error Rate is defined as :
        EER = (Conf + Miss + FA) / (Hit + Conf + Miss)

    Arguments:
        ref ... a list of dicts indexed by utt_id with reference events
        hyp ... a list of dicts indexed by utt_id with hypothesis events
        events ...  maps event labels (str) to symbols (int)
        thr ... threshold for detection (def:float = 0)
    Return:
        {'score':[H/M/FA/C], 'EER':EER}
    """
    # Checks
    for item in [ref,hyp,events]:
        if not item:
            print(f'Error compute_eer(): {item} is empty.')
            exit(1)
    for item in [ref,hyp]:
        if not isinstance(item,list):
            print(f'Error compute_eer(): {item} is not a list.')
            exit(1)
    if not isinstance(events,dict):
        print(f'Error compute_eer(): events is not a dict.')
        exit(1)

    # Compute cost matrix, negative costs (no-overlap) are set to 0
    (reflen,hyplen) = len(ref),len(hyp)
    C = np.zeros(shape=(reflen,hyplen),dtype=np.float32)
    I = np.zeros(shape=(reflen,hyplen),dtype=np.bool)

    for i,ref_event in enumerate(ref):
        for j,hyp_event in enumerate(hyp):       
            (ref_on,ref_dur,ref_lbl) = ref_event['onset'],ref_event['duration'],ref_event['label']
            (hyp_on,hyp_dur,hyp_lbl) = hyp_event['onset'],hyp_event['duration'],hyp_event['label']
            o = max(0, min(ref_on+ref_dur, hyp_on+hyp_dur) - max(ref_on,hyp_on))
            C[i,j] = o / (ref_dur + hyp_dur - o)
            I[i,j] = (events[ref_lbl] == events[hyp_lbl])

    # Identify contiguous regions, these are candidates for hits/confusions
    # Find optimal pairs for each region
    cand = list()
    (regC, reg_num) = ndimage.label(C)
    for lbl in np.arange(1, reg_num+1):
        reg_idx = list(np.argwhere(regC == lbl))

        # Iterative find/remove items from reg_idx
        if len(reg_idx) < 2:
            optim = tuple(reg_idx[0])
            cand.append((optim,C[optim]))
        else:
            while len(reg_idx) > 0:
                (optim, val) = 0, 0.0
                for pair in reg_idx:
                    if C[tuple(pair)] > val:
                        optim = tuple(pair)
                        val = C[optim]
                cand.append((optim,val))
                reg_idx = [idx for idx in reg_idx if not (idx[0]==optim[0] or idx[1]==optim[1])]

    # Compute EER, generate ali
    ali = list()
    (h,c) = (0,0)
    for (pair,val) in cand:
        if val > thr:
            ali.append((pair,val))
            if I[pair]:
                h += 1
            else:
                c += 1               
    m = reflen - h - c
    fa = hyplen - h - c

    score = np.array([h,m,fa,c],dtype=np.float32)
    eer = round(100*(c+m+fa)/reflen,2)
    return ({'score':score, 'EER':eer})

