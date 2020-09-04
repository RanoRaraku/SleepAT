"""
Made by Michal Borsky, 2019, copyright (C) RU
Compute detection error from annotation, transcription, and alignment
"""
import numpy as np

def compute_der(annot:dict, trans:dict, align:dict, events:dict, threshold:float=0.5) -> tuple:
    """
    Compute localization and classification errors for a detector. Inputs are 'annot', 'trans',
    and 'align' dicts which use the same utt_id keys. One can supply a threshold for positive
    detection. 'Events' dict and 'exclude_event' can be supplied to exclude 'event' type from
    error summary. Regarding classification, an event is correctly (C) detected if metric >=
    'threshold' and the labels match. Misses consist of insertions (I), substitutions (S), and
    deletions (D). The definitions are as:
    C if (AoO >= thr.) && (ref == hyp)
    S if (AoO >= thr.) && (ref != hyp)
    I if (AoO < thr.) && (ref != hyp)
    D if (AoO < thr.) && (ref == hyp)
    These flags are added to 'align' and the updated dict is returned.

    The localization error assesses boundary placement, beginning and end, for correctly detected
    events only. Values are mean absolute deltas in time units the events use. Formulas:
    beg_prec = mean(|ref_beg - hyp_beg|)
    end_prec = mean(|ref_end - hyp_end|)

    The classification error is expressed as precision and recall for each class in events
    separatelly.

    Arguments:
        annot ... a dict with utt_ids with annotated events
        trans ... a dict with utt_ids with predicted events
        align ... a dict with utt_ids with alingment betweet annot and trans
        events ...  maps event text labels to numerical values
        threshold ... alignment metric threshold for positive detection (def:float = 0.5)
    Return:
        updated aligned_scorings with decisions and stats dictionary

    ToDo:
        check for empty ref/hyp/ali
    """
    # Checks
    required = [annot, trans, align, events]
    for item in required:
        if not item:
            print(f'Error: {item} is empty.')
            exit(1)
    if threshold < 0:
        print(f'Error: set threshold is < 0 ({threshold}).')
        exit(1)

    events_inv = dict()
    for key,val in events.items():
        if val in events_inv and key != 'null':
            continue
        events_inv[val] = key        
    enum = len(events_inv.keys())

    # Accumulate stats, use a value indexing instead of if/else
    # ISDC indexing is: 0->I, 1->S, 2->D, 3->C, add text value to original align
    isdc = [0,0,0,0]
    (beg_mad,end_mad) = 0.0,0.0
    val2txt = {0:'I',1:'S',2:'D',3:'C'}
    conf_mat = np.zeros(shape=(enum,enum),dtype=np.uint32)

    for utt_id,ali in align.items():
        (ref,hyp) = annot[utt_id], trans[utt_id]
        for pair in ali:
            score = pair['Score']
            (ref_event, hyp_event) = ref[pair['Ref']],hyp[pair['Hyp']]
            (oref, dref, lref) = ref_event['onset'], ref_event['duration'], ref_event['label']
            (ohyp, dhyp, lhyp) = hyp_event['onset'], hyp_event['duration'], hyp_event['label']  

            # Conf_mat population
            (i,j) = events[lref], events[lhyp]
            conf_mat[i,j] += 1

            # ISCD flag
            val = (score >= threshold) + 2*(lref == lhyp)
            isdc[val] += 1
            if val == 3:
                beg_mad += abs(oref - ohyp)
                end_mad += abs(oref + dref - ohyp - dhyp)
            pair['ISDC'] = val2txt[val]

    # Compute classify + boundary errors
    if isdc[3] == 0:
        (beg_mad,end_mad) = [float('nan'),float('nan')]
    else:
        beg_mad = round(beg_mad/isdc[3],6)
        end_mad = round(end_mad/isdc[3],6)

    classify = dict()
    for i in range(enum):
        tp = conf_mat[i,i]
        prec = tp / conf_mat[:,i].sum()
        rec = tp / conf_mat[i,:].sum()

        classify[events_inv[i]] = {'prec':prec, 'rec':rec}

    return (align, {'ISDC':isdc,'Beg_MAD':beg_mad,'End_MAD':end_mad})
