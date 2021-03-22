"""
Made by Michal Borsky, 2020, copyright (C) RU.
Accumulate per utterance scores [H,M,FA,C] on per-subject or total basis.
"""
import numpy as np

def accumulate_score(per_utt:dict, mode:str='total', sub2utt:dict = None) -> dict:
    """
    Accumulate scores from compute_ser() or compute_der() on per_subject or total basis.
    The per_utt are a dict formatted as {utt_id: [H,M,FA,C]}, where utt_id is the utterance
    identifier and score is [H,M,FA,C] list. The mode determines the accumulation style, can
    be either on per_subject or total basis. The sub2utt dictionary is needed for "per_sub"
    accumulation mdoe. The output is a dictionary formatted as {'id': score}, where "id" is
    either a "spk_id" or "total" string. The score is again a [H,M,FA,C] list.

    Arguments:
        per_utt ... a dict with a score for each utterance, i.e. {utt_id: [H,M,FA,C]}
        mode ... an accumulation mode <total|per_sub>, (default:str = 'total')
        sub2utt ... a dict that maps each utterance to a speaker (default:dict = None)
    Return:
        out ... a dictionary in the form {id: score}
    """
    # Checks
    if not isinstance(per_utt,dict):
        print(f'Error accumulate_score(): per_utt is not a dict.')
        exit(1)            
    if not per_utt:
        print(f'Error accumulate_score(): per_utt is empty.')
        exit(1)

    if mode == 'total':
        sub2utt = {'total': list(per_utt.keys()) }
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
        acc = np.zeros_like(per_utt[utt_lst[0]])
        for utt in utt_lst:
            acc += per_utt[utt]
        out[sub] = acc

    return out
