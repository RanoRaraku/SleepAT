"""
Made by Michal Borsky, 2020, copyright (C) RU.
Accumulate per recerance scores [H,M,FA,C] on per-subject or total basis.
"""
import numpy as np

def accumulate_score(per_rec:dict, mode:str='total', sub2rec:dict = None) -> dict:
    """
    Accumulate scores from compute_ser() or compute_der() on per_subject or total basis.
    The per_rec are a dict formatted as {rec_id: [H,M,FA,C]}, where rec_id is the recerance
    identifier and score is [H,M,FA,C] list. The mode determines the accumulation style, can
    be either on per_subject or total basis. The sub2rec dictionary is needed for "per_sub"
    accumulation mdoe. The output is a dictionary formatted as {'id': score}, where "id" is
    either a "spk_id" or "total" string. The score is again a [H,M,FA,C] list.

    Arguments:
        per_rec ... a dict with a score for each recerance, i.e. {rec_id: [H,M,FA,C]}
        mode ... an accumulation mode <total|per_sub>, (default:str = 'total')
        sub2rec ... a dict that maps each recerance to a speaker (default:dict = None)
    Return:
        out ... a dictionary in the form {id: score}
    """
    # Checks
    if not isinstance(per_rec,dict):
        print(f'Error accumulate_score(): per_rec is not a dict.')
        exit(1)            
    if not per_rec:
        print(f'Error accumulate_score(): per_rec is empty.')
        exit(1)

    if mode == 'total':
        sub2rec = {'total': list(per_rec.keys()) }
    elif mode == 'per_sub':
        if not isinstance(sub2rec,dict):
            print(f'Error accumulate_score(): sub2rec is not a dict.')
            exit(1)            
        if not sub2rec:
            print(f'Error accumulate_score(): sub2rec is empty.')
            exit(1)
    else:
        print(f'Error accumulate_score(): unrecognized mode {(mode)}.')
        exit(1)        

    # Init stats and accumulate
    out = dict()
    for sub, rec_lst in sub2rec.items():
        acc = np.zeros_like(per_rec[rec_lst[0]])
        for rec in rec_lst:
            acc += per_rec[rec]
        out[sub] = acc

    return out
