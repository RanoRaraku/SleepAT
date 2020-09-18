"""
Made by Michal Borsky, 2019, copyright (C) RU
Compute detection error between a reference and hypothesis given their alignment
"""
import numpy as np

def compute_der(ref:list, hyp:list, ali:list, events:dict, thr:float=2/3) -> tuple:
    """
    Compute localization and classification errors for a detector. Inputs are 'ref', 'hyp',
    and 'ali' dicts which use the same utt_id keys. One can supply a threshold for positive
    detection. 'Events' dict and 'exclude_event' can be supplied to exclude 'event' type from
    error summary. Regarding classification, an event is correctly (C) detected if metric >=
    'threshold' and the labels match. Misses consist of substitutions (S), insertions (I), and
    deletions (D). The definitions are as:
    C if 
    S if 
    I if 
    D if
    These flags are added to 'ali' and the updated dict is returned. The classification
    error is expressed as precision and recall for each event class separatelly. Null events
    are omitted by default. The return values are a summary across all utterances in 
    ref/hyp/ali. 'Null' events are excluded from evaluation.

    The localization error assesses boundary placement (beg/end), for correctly detected
    events only. Values are mean absolute deltas in time units the events use. Formulas:
    beg_prec = mean(|ref_beg - hyp_beg|)
    end_prec = mean(|ref_end - hyp_end|)

    Arguments:
        ref ... a dict with utt_ids with refated events
        hyp ... a dict with utt_ids with predicted events
        ali ... a dict with utt_ids with alingment betweet ref and hyp
        events ...  maps event text labels to numerical values
        threshold ... metric threshold for positive detection (def:float = 2/3)
    Return:
        tuple of (ali updated with decisions, results)
    ToDo
        try i,j,k indexing
        remove duplicate items from alignmenet        
    """
    # Checks
    required = [ref, hyp, ali, events]
    for item in required:
        if not item:
            print(f'Error: {item} is empty.')
            exit(1)
    if thr < 0:
        print(f'Error: thr is < 0 ({thr}).')
        exit(1)

    # Accumulate stats
    sym2int = {'H':0, 'M':1, 'I':2, 'S':3}
    hmis = [0,0,0,0]
    for pair in ali:
        (score,ref_idx,hyp_idx) = (pair['score'],pair['ref_idx'],pair['hyp_idx'])
        if score >= thr:
            (lref,lhyp) = (ref[ref_idx]['label'],hyp[hyp_idx]['label'])
            if lref == lhyp:
                flag = 'H'
            else:
                flag = 'S'
        else:
            argmax = pair['argmax']
            if argmax == 'ref':
                flag = 'I'
            else:
               flag = 'M'
        pair['HMIS'] = flag
        hmis[sym2int[flag]] += 1

    (H,M,I,S) = hmis
    der = round( 100*(M+I+S)/(H/2+M+S),2)
    return (ali,hmis,der)