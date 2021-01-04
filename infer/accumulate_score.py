"""
Made by Michal Borsky, 2020, copyright (C) RU.
Compute standard performance measures.
"""
import numpy as np

def accumulate_score(scores:dict, mode:str='total', sub2utt:dict = None) -> dict:
    """
    Accumulate scores from compute_ser() or compute_der() on per_subject or global

    Arguments:
        scores ...
        sub2utt ...
        mode ...
    Return:
        score ...
    """
    # Checks
    if not isinstance(scores,dict):
        print(f'Error accumulate_score(): scores is not a dict.')
        exit(1)            
    if not scores:
        print(f'Error accumulate_score(): scores is empty.')
        exit(1)

    if mode == 'total':
        sub2utt = {'total': list(scores.keys()) }
    elif mode == 'per_sub':
        if not isinstance(sub2utt,dict):
            print(f'Error accumulate_score(): sub2utt is not a dict.')
            exit(1)            
        if not sub2utt:
            print(f'Error accumulate_score(): sub2utt is empty.')
            exit(1)
    else:
        print(f'Error accumulate_score(): unrecognized mode {(mode)}.')
        exit(1)        

    # Init stats and accumulate
    out = dict()
    for sub, utt_lst in sub2utt.items():
        acc = np.zeros_like(scores[utt_lst[0]])
        for utt in utt_lst:
            acc += scores[utt]
        out[sub] = acc

    return out
