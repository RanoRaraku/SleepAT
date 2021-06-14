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

    for k,re in enumerate(ref):
        for l,he in enumerate(hyp):
            (re_on, re_dur) = re['onset'],re['duration']
            (he_on, he_dur) = he['onset'],he['duration']
            o = max(0, min(re_on+re_dur, he_on+he_dur) - max(re_on,he_on))
            C[k,l] =  2*o / (re_dur + he_dur)
            I[k,l] = (events[re['label']] == events[he['label']])

    # Identify contiguous regions, these are candidates for hits/confusions
    # Find optimal pairs for each region
    cnd = list()
    (image, reg_num) = ndimage.label(C)
    for lbl in np.arange(1, reg_num+1):
        kl_lst = list(np.argwhere(image == lbl))

        # Iterative find/remove items from kl_lst
        if len(kl_lst) < 2:
            optim = tuple(kl_lst[0])
            cnd.append((optim,C[optim]))
        else:
            while len(kl_lst) > 0:
                (optim, val) = (-1,-1), 0.0
                for pair in kl_lst:
                    if C[tuple(pair)] > val:
                        optim = tuple(pair)
                        val = C[optim]
                cnd.append((optim,val))
                kl_lst = [i for i in kl_lst if not (i[0]==optim[0] or i[1]==optim[1])]

    # Assign labels, generate alingment
    ali = list()
    (h,c) = (0,0)
    for (pair,val) in cnd:
        if val > thr:
            ali.append((pair,val))
            if I[pair]:
                h += 1
            else:
                c += 1               
    m = reflen - h - c
    fa = hyplen - h - c
    score = np.array([h,m,fa,c],dtype=np.uint32)
 
    return score
