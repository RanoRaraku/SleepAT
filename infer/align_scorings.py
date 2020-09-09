"""
Made by Michal Borsky, 2019, copyright (C) RU
Align ref and hyp for evaluation.
"""
import numpy as np
from matplotlib import pyplot as plt

def align_scorings(ref:list, hyp:list) -> list:
    """
    Aligns 'ref' and 'hyp' for an utterance. The inputs are lists of scored & predicted
    events, each event being a dict() with boundary marks. Lists can't be empty, but they
    can be of unequal length. The alignemnt metric is the length-of-overlap (LoO), computed
    as (2*t_overlap / t_ref+t_hyp), 't' being duration. LoO is from real interval (-Inf, 1>,
    where negative means a displacement between ref and hyp. Events are aligned using dynamic
    programming to find optimal event pairs, meaning LoO is maximized with a minimum number of
    steps. Negative LoO are needed to ensure the minimum condition. The output is a list of
    dicts containing {ref_id, hyp_id, LoO value), where 'id' is the index. The alignemnt can
    be used with ref & hyp only, as events are only referenced by their indexes.

    Arguments:
        ref ... ref scoring as list of events(dicts)
        hyp ... hyp scoring as a list of events(dicts)
    Return:
        ali ...list of dicts with in form of (ref_id, hyp_id, LoO value)

    ToDo: rework to start DP from beg to have it ordered
    """
    # Check for empty ref/hyp
    for scoring in [ref,hyp]:
        if not scoring:
            print(f'Error: {scoring} is empty, nothing to align.')
            exit(1)

    # Compute LoO matrix
    (reflen,hyplen) = len(ref),len(hyp)
    LoO = np.zeros(shape=(reflen,hyplen),dtype=np.float32)

    for i,ref_event in enumerate(ref):
        for j,hyp_event in enumerate(hyp):
            (oref,dref) = ref_event['onset'],ref_event['duration']
            (ohyp,dhyp) = hyp_event['onset'],hyp_event['duration']
            num = min(oref + dref, ohyp + dhyp) - max(oref, ohyp)
            LoO[i,j] = 2*num/(dref + dhyp)

    # Find max-LoO path, that is optimal alignment
    # Avoid ambiguous pathing to -Inf by stopping at (1,1)
    path = list()
    (i,j) = (reflen,hyplen) # start from end, account for (+1,+1) offset to LoO_ext
    LoO_ext = np.ones(shape=(reflen+1,hyplen+1))*np.NINF
    LoO_ext[1:,1:] = LoO
    while (i,j) != (1,1):
        path.append((i-1,j-1))  # remove (+1,+1) offset
        (min_i,min_j) = max((i-1,j),(i,j-1),(i-1,j-1), key=lambda x:LoO_ext[x])
        (i,j) = (min_i,min_j)
    path.append((i-1,j-1))  # push the last step

    # Push everything into a list of {ref_id, hyp_id, LoO_value} dicts
    ali = list()
    for (i,j) in path:
        ali.append({'Ref':i,'Hyp':j,'Score': round(LoO[i,j],3)})

    return ali
