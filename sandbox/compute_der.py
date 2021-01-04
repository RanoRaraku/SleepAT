"""
Made by Michal Borsky, 2019, copyright (C) RU
Compute detection error rate between reference and hypothesis.
"""
import numpy as np
import sleepat 
from sleepat import io

def compute_der(ref:list, hyp:list, events:dict, thr:float=1/2) -> dict:
    """
    Compute dvent error rate (DER) at an event level. Inputs are 'ref', 'hyp', which are 
    lists of dicts, and 'events' which is a dict. Each event in hyp/ref  has 'onset', duration',
    and 'label' keys defined, i.e. {'label':snore, 'onset':0.0, 'duration': 1.0}. Events dict
    maps between event labels (string) event symbols (int) i.e. {'snore':1}. All event types
    in ref and hyp are included in evaluation, it might be prudent to remove some that are not
    to be scored before. Ref/hyp/events can't be empty, but they can be of unequal length.
    The implementation assumes events are chronologically ordered and don't overlap.    
    
    We align ref/hyp using relative length of overlap (LoO),
    record (ref_i), (hyp_j) indexes, LoO value, and what event it was. The decision labels:
    Hit        : LoO >= thr and labels match
    Confusion  : LoO >= thr and labels match
    Miss       : LoO < thr and argmax == hyp
    False Alarm: LoO < thr and argmax == ref

    The Event Error Rate is defined as :
        EER = (Conf + Miss + FA) / (Hit + Conf Miss)

    Arguments:
        ref ... a list of dicts indexed by utt_id with reference events
        hyp ... a list of dicts indexed by utt_id with hypothesis events
        events ...  maps event labels (str) to symbols (int)
        thr ... threshold value for successfull detection (default:float = 2/3)
    Return:
        dict {'score':[H/M/FA/C], 'DER':DER}
    """
    # Checks
    for item in [ref,hyp,events]:
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
        print(f'Error compute_der(): thr is < 0 ({thr}).')
        exit(1)
    elif thr < 1/2:
        print(f'Warning compute_der(): thr is < 1/2 ({thr}). This is not advised.')
    thr = round(thr,4)


    # Compute cost matrix, negative costs are set to 0, indicating no overlap
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

    # Find the closest event for every ref/hyp and record it
    # Version with full tracing and checks, but inefficient
    # Sanity checks:
        # 1) Argmax progressively drops search candidates, needs min(LoO) = 0
        # 2) If only 0 exist (no overlap), no pairing i/j event is recorded (NaN)
    # Warning: It relies on np.argmax() to return 1st arg. if multiple exist!
    (k, l, ali) = (0, 0, list())
    for i in range(reflen):
        j = C[i,k:].argmax() + k
        (k, val) = (j, round(C[i,j],8))
        if val == 0:
            j = float('nan')
        ali.append({'argmax':'hyp', 'pair':(i,j), 'cost':val})

    for j in range(hyplen):
        i = C[l:,j].argmax() + l
        (l, val) = (i, round(C[i,j],8))
        if val == 0:
            i = float('nan')
        ali.append({'argmax':'ref','pair':(i,j), 'cost':val})

    # Score alignment
    # Rely on exclusive behavior of C[i,j], if cost > 2/3 then it can only be hit
    # There also is another item in ali with the same ref_idx, hyp_idx, cost
    (h,m,fa,c) = (0,0,0,0)
    for item in ali:
        if item['cost'] > thr:
            if I[item['pair']]:
                item['H/M/FA/C'] = 'H'
                h += 0.5
            else:
                item['H/M/FA/C'] = 'C'
                c += 0.5
        else:
            argmax =item['argmax']
            if argmax == 'ref':
                item['H/M/FA/C'] = 'FA'
                fa += 1
            else:
                item['H/M/FA/C'] = 'M'
                m += 1

    score = np.array([h,m,fa,c],dtype=np.int32)
    der = round(100*(c+m+fa)/reflen,2)

    print(score,der)

    return ({'score':score, 'DER':der})