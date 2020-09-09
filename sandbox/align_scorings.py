"""
Made by Michal Borsky, 2019, copyright (C) RU
Align ref and hyp for evaluation.
"""
import numpy as np

def align_scorings(ref:list, hyp:list) -> list:
    """
    Aligns 'ref' and 'hyp' for an utterance. The inputs are lists of scored & predicted
    events, each event being a dict() with boundary marks. Lists can't be empty, events
    need to be continuous, meaning end of event_[n] equals beg. of event_[n+1], and the
    events span the same time-frame. The alignemnt metric is the area-of-overlap (AoO),
    computed as (2*t_overlap / t_ref+t_hyp), 't' being duration. Events are aligned using
    dynamic programming to find optimal event pairs, meaning AoO is maximized. The return
    is a list of dicts containing {ref_id, hyp_id, AoO value), where 'id' is the index.
    The alignemnt can be used with ref & hyp only, as events are only referenced.

    Arguments:
        ref ... ref scoring as list of events(dicts)
        hyp ... hyp scoring as a list of events(dicts)
    Return:
        ali ...list of dicts with in form of (ref_id, hyp_id, AoO value)
    """
    # Check for empty ref/hyp
    # Check if ref/hyp are fully populated & of equal duration
    dur = list()
    for scoring in [ref,hyp]:
        if not scoring:
            print(f'Error: {scoring} is empty.')
            exit(1)

    for scoring in [ref,hyp]:
        onset = 0.0
        for event in scoring:
            if event['onset'] != onset:
                print(f'Error: events in {scoring} have a gap, use utils.normalize_scoring().')
                exit(1)
            onset = round(onset + event['duration'],6)
        dur.append(onset)

    if dur[0] != dur[1]:
        print(f'Error: ref. and hyp. have different durations ({dur[0]} vs. {dur[1]}).')
        exit(1)

    # Compute AoO matrix
    (reflen,hyplen) = len(ref),len(hyp)
    AoO = np.zeros(shape=(reflen,hyplen),dtype=np.float32)

    for i,ref_event in enumerate(ref):
        for j,hyp_event in enumerate(hyp):
            (refon,refdur) = ref_event['onset'],ref_event['duration']
            (hypon,hypdur) = hyp_event['onset'],hyp_event['duration']
            num = min(refon + refdur, hypon + hypdur) - max(refon, hypon)
            AoO[i,j] = 2*num/(refdur + hypdur)

    # Find max-AoO path, that is optim alignment
    # Avoid ambiguous pathing to -Inf by stopping at (1,1)
    path = list()
    (i,j) = (reflen,hyplen) # account for (+1,+1) offset to AoO_ext
    AoO_ext = np.ones(shape=(reflen+1,hyplen+1))*np.NINF
    AoO_ext[1:,1:] = AoO
    while (i,j) != (1,1):
        path.append((i-1,j-1))  # remove (+1,+1) offset
        (min_i,min_j) = max((i-1,j),(i,j-1),(i-1,j-1), key=lambda x:AoO_ext[x])
        (i,j) = (min_i,min_j)
    path.append((i-1,j-1))

    # Push everything into a list of {ref_id,hyp_id,AoO_value} dicts
    ali = list()
    for (i,j) in path:
        ali.append({'Ref':i,'Hyp':j,'Score': round(AoO[i,j],3)})

    return ali
