"""
Made by Michal Borsky, 2020, under Apache License v2.0 license.
Evaluate identification/detection error.
"""
import numpy as np
import scipy
from scipy import ndimage

def eval_der(ref:list, hyp:list, events:dict, thr:float = 2/3) -> np.ndarray:
    """
    Evaluate detection error at a discrete level. Inputs are 'ref', 'hyp', which are
    lists, and 'events' which is a dict. Each event in hyp/ref  has 'onset', duration',
    and 'label' keys defined, i.e. {'label':snore, 'onset':0.0, 'duration': 1.0}. 'Events'
    maps between event labels (string) and event symbols (int) i.e. {'snore':1}. Used in case
    multiple labels (central/obstructive apnea) map to the same symbol. Threshold
    can be used to set a minimal value for a successful detection, thr = 0 means any
    non-zero overlap is a hit. All events in ref and hyp are included in evaluation,
    remove events that are not to be scored beforehand. Ref/events can't be empty.
    The implementation assumes events are chronologically ordered and don't overlap.

    Calculate relative overlap (C) between ref-hyp, find an optimal alignment, and assign decision
    labels defined as:
    Hit        : C > trh && event labels match
    Confusion  : C > trh && event labels don't match
    Miss       : C <= trh && event is from ref
    False Alarm: C <= trh && event is from hyp

    Arguments:
        ref ... a list of reference events, each event is a dictionary
        hyp ... a list of hypothesis events, each event is a dictionary
        events ...  maps event labels (str) to symbols (int)
        thr ... threshold for successful detection (def:float = 2/3)
    Return:
        score as numpy array (h/m/fa/c)
    """
    # Checks
    for item in [ref,events]:
        if not item:
            print(f'Error eval_der(): {item} is empty.')
            exit(1)
    for item in [ref,hyp]:
        if not isinstance(item,list):
            print(f'Error eval_der(): {item} is not a list.')
            exit(1)
    if not isinstance(events,dict):
        print(f'Error eval_der(): events is not a dict.')
        exit(1)
    if thr < 0:
        print(f'Error eval_der(): thr < 0.')
        exit(1)
    if not hyp:
        return np.array([0,len(ref),0,0],dtype=np.uint32)


    # Compute cost/identity matrix, negative costs (no-overlap) are set to 0
    (reflen,hyplen) = len(ref),len(hyp)
    C = np.zeros(shape=(reflen,hyplen),dtype=np.float32)
    I = np.zeros(shape=(reflen,hyplen),dtype=np.bool)

    # Iterate over reference/hypothesis events (re/he) and calculate cost/identity
    for k,re in enumerate(ref):
        for l,he in enumerate(hyp):
            (re_on, re_dur) = re['onset'],re['duration']
            (he_on, he_dur) = he['onset'],he['duration']
            o = max(0, min(re_on+re_dur, he_on+he_dur) - max(re_on,he_on))
            C[k,l] =  2*o / (re_dur + he_dur)
            I[k,l] = (events[re['label']] == events[he['label']])

    # Find contiguous regions of non-zero values, these are candidates (cnd) for hits/confusions
    # region is a list of (k,l)-indexes with respect to C matrix
    cnd = list()
    (image, region_num) = ndimage.label(C)
    for lbl in np.arange(1, region_num+1):
        region = list(np.argwhere(image == lbl))

        # Find optimal (k,l)-pairs, remove competing pairs
        if len(region) < 2:
            optim = tuple(region[0])
            cnd.append((optim,C[optim]))
        else:
            while len(region) > 0:
                (optim, val) = (-1,-1), 0.0
                for pair in region:
                    if C[tuple(pair)] > val:
                        optim = tuple(pair)
                        val = C[optim]
                cnd.append((optim,val))
                region = [i for i in region if not (i[0]==optim[0] or i[1]==optim[1])]

    # Assign H/C labels, generate alignment, keep track of aligned k,l
    (h,c) = (0,0)
    (ali, ali_k, ali_l) = list(), list(), list()
    for (pair,val) in cnd:
        if val > thr:
            ali_k.append(pair[0])
            ali_l.append(pair[1])

            if I[pair]:
                h += 1
                ali.append((pair, val, 'H'))
            else:
                c += 1
                ali.append((pair, val, 'C'))

    # Assign M/FA labels based on events missing in ali
    (m,fa) = (0,0)
    for k in set(range(0,reflen)).difference(ali_k):
        m +=1
        ali.append(((k,float('nan')), max(C[k,:]), 'M') )

    for l in set(range(0,hyplen)).difference(ali_l):
        fa +=1
        ali.append(((float('nan'),l), max(C[:,l]), 'FA'))

    score = np.array([h,m,fa,c],dtype=np.uint32)
    return (score, ali)
