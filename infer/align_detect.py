"""
Made by Michal Borsky, 2019, copyright (C) RU
Align ref and hyp for evaluation.
"""
import numpy as np


def align_detect(ref:list, hyp:list) -> list:
    """
    Align reference and hypothesi for detection. Detection is understood as localization and
    classification. This script only deals with localization part, the classification is done
    in compute_der(), i.e. The inputs are lists of reference & hypothesis events, each event
    being a dict with boundary marks. Lists can't be empty, but they can be of unequal length.
    The script is called for each utterance separately, not whole annot/trans files. Localize
    metric is a relative length of overlap (LoO), values being from (-Inf, 1> interval, where
    negative means a displacement. A maximum is then found along each LoO axis, meaning for
    each ref/hyp we find the closest hyp/ref.event separately. The output is combined list of
    pairs, logged as {argmax, ref_idx, hyp_idx, LoO value}. 'Argmax' indicates along which axis
    the maxima were found, 'ref_idx'/'hyp_idx' reference event index, and 'LoO' value.

    Given 2 events, ref and hyp, and their beginning and end timestamps, LoO is computed as:
    t_overlap = [min(ref_end,hyp_end) - max(ref_beg,hyp_beg)]
    LoO[ref,hyp] = 2*t_overlap / [(ref_end-ref_beg)  + (hyp_end-hyp_beg)]

    Warning: There can be duplicates in list, these usually mean a correctly placed ref/hyp.
    The alignemnt can only be used with ref & hyp, as events are only referenced by their indexes.

    Arguments:
        ref ... reference scoring containing a list of events, each dict()
        hyp ... hypothesis scoring containing a list of events, each a dict()
    Return:
        ali ...list of dicts in form of {argmax, ref_idx, hyp_idx, LoO value}
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
            LoO[i,j] = 2*num/(dref+dhyp)

    # Find the closest event for every ref/hyp and log it int
    ali = list()
    for i,j in enumerate(LoO.argmax(axis=1)):
        ali.append({'argmax':'hyp','ref_idx':i, 'hyp_idx':j, 'score':float(LoO[i,j])})    
    for j,i in enumerate(LoO.argmax(axis=0)):
        ali.append({'argmax':'ref','ref_idx':i, 'hyp_idx':j, 'score':float(LoO[i,j])})

    return ali

