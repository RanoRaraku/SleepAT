"""
Made by Michal Borsky, 2019, copyright (C) RU
Compute detection error from annotation, transcription, and alignment
"""
import numpy as np

def compute_der(annot:dict, trans:dict, align:dict, events:dict, threshold:float=0.5,
    exclude_null:bool = True) -> tuple:
    """
    Compute localization and classification errors for a detector. Inputs are 'annot', 'trans',
    and 'align' dicts which use the same utt_id keys. One can supply a threshold for positive
    detection. 'Events' dict and 'exclude_event' can be supplied to exclude 'event' type from
    error summary. Regarding classification, an event is correctly (C) detected if metric >=
    'threshold' and the labels match. Misses consist of insertions (I), substitutions (S), and
    deletions (D). The definitions are as:
    C if (LoO >= thr.) && (ref == hyp)
    S if (LoO >= thr.) && (ref != hyp)
    I if (LoO < thr.) && (ref != hyp)
    D if (LoO < thr.) && (ref == hyp)
    These flags are added to 'align' and the updated dict is returned. The classification error
    is expressed as precision and recall for each event class separatelly. Null events are omitted
    by default.

    The localization error assesses boundary placement, beginning and end, for correctly detected
    events only. Values are mean absolute deltas in time units the events use. Formulas:
    beg_prec = mean(|ref_beg - hyp_beg|)
    end_prec = mean(|ref_end - hyp_end|)

    Arguments:
        annot ... a dict with utt_ids with annotated events
        trans ... a dict with utt_ids with predicted events
        align ... a dict with utt_ids with alingment betweet annot and trans
        events ...  maps event text labels to numerical values
        threshold ... alignment metric threshold for positive detection (def:float = 0.5)
    Return:
        tuple of (align updated with decisions, results)
    """
    # Checks
    required = [annot, trans, align, events]
    for item in required:
        if not item:
            print(f'Error: {item} is empty.')
            exit(1)
    if threshold < 0:
        print(f'Error: supplied threshold is < 0 ({threshold}).')
        exit(1)




    # Accumulate stats, use a i,j,k indexing instead of if/else
    # iscd dims (i,j,k): i-> thr (</>=), j-> match (N/Y), k-> ref event class
    # bound dims (l,k): l-> boundary (beg/end), k-> ref event class
    events_num = len(set(events.values()))    
    bounds = np.zeros(shape=(2,events_num),dtype=np.float32)
    isdc = np.zeros(shape=(2,2,events_num),dtype=np.uint32)
    sym2int = {0:'I',1:'S',2:'D',3:'C'}

    for utt_id, ali in align.items():
        (ref,hyp) = (annot[utt_id], trans[utt_id])
        for pair in ali:
            score = pair['Score']
            (ref_event, hyp_event) = ref[pair['Ref']],hyp[pair['Hyp']]
            (oref, dref, lref) = ref_event['onset'], ref_event['duration'], ref_event['label']
            (ohyp, dhyp, lhyp) = hyp_event['onset'], hyp_event['duration'], hyp_event['label']

            # bound/csid population
            (i,j,k) = ((score >= threshold)*1, (lref == lhyp)*1, events[lref])
            isdc[i,j,k] += 1
            bounds[:,k] += [abs(oref-ohyp), abs(oref+dref-ohyp-dhyp)]
            pair['isdc'] = sym2int[i*1+j*2]

    # Populate results dictionary for each 'events' label
    results = dict()
    for key,val in events.items():
        if exclude_null and val == events['null']:
            continue

        C = isdc[1,1,val]
        acc = C/isdc[:,:,val].sum()
        (prec, rec) = C/isdc[1,:,val].sum(), C/isdc[:,1,val].sum()
        (beg_mad, end_mad) = round(bounds[0,val]/C ,6), round(bounds[1,val]/C ,6)
        results[key] = {'isdc':isdc[:,:,val].flatten(), 'accuracy':acc, 'precision':prec,
            'recall':rec, 'beg_mad':beg_mad, 'end_mad':end_mad}

    return (align, results)
