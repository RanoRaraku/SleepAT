"""
Made by Michal Borsky, 2020, copyright (C) RU
Compute identification/detection error rate between reference and hypothesis.
"""
import numpy as np
import scipy
from scipy import ndimage

def compute_der(ref:list, hyp:list, events:dict, thr:float = 2/3) -> dict:
    """
    Compute detection error at an event level. Inputs are 'ref', 'hyp', which are
    lists, 'events' which is a dict. Each event in hyp/ref  has 'onset', duration',
    and 'label' keys defined, i.e. {'label':snore, 'onset':0.0, 'duration': 1.0}. Events
     maps between event labels (string) event symbols (int) i.e. {'snore':1}. Threshold
    can be used to set a minimal value for successful detection, thr = 0 means any 
    non-zero overlap is a hit. All event types in ref and hyp are included in evaluation,
    remove events that are not to be scored before. Ref/hyp/events can't be empty, but
    they can be of unequal length. The implementation assumes events are chronologically
    ordered and don't overlap.

    Align ref/hyp using relative length of overlap (LoO), record (ref_i), (hyp_j) indexes,
    LoO value. The decision labels:
    Hit        : LoO > trh && event labels match
    Confusion  : LoO > trh && event labels don't match
    Miss       : LoO < trh && event is from ref
    False Alarm: LoO < trh && event is from hyp
    
    Arguments:
        ref ... a list of dicts indexed by utt_id with reference events
        hyp ... a list of dicts indexed by utt_id with hypothesis events
        events ...  maps event labels (str) to symbols (int)
        thr ... threshold for successful detection (def:float = 2/3)
    Return:
        score as numpy array (H/M/FA/C)
    """
    # Checks
    for item in [ref,events]:
        if not item:
            print(f'Error compute_der(): {item} is empty.')
            exit(1)
    for item in [ref,hyp]:
        if not isinstance(item,list):
            print(f'Error compute_der(): {item} is not a list.')
            exit(1)
    if not isinstance(events,dict):
        print(f'Error compute_der(): events is not a dict.')
        exit(1)
    if thr < 0:
        print(f'Error compute_der(): thr < 0.')
        exit(1)
    if not hyp:
        return np.array([0,len(ref),0,0],dtype=np.uint32)



    # Compute cost matrix, negative costs (no-overlap) are set to 0
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

    # Compute error, generate ali
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