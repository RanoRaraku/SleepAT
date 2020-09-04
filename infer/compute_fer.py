"""
Made by Michal Borsky, 2019, copyright (C) RU
Compute frame error rate (FER).
"""
import numpy as np
import sleepat
from sleepat import io

def compute_fer(targets:dict, posteriors:dict, events:dict) -> tuple:
    """
    Compute frame error rate between reference and hypothesis. The inputs are dicts targets.scp
    and post.scp. We assume these two are indexed by utt_id, frame-aligned, of equal length. The
    posteriors are assigned class using MAP (maximum a posterior) rule. 'Events' can be supplied
    to produce a labelled confusion matrix, otherwise, the labels correspond to 
    
    The computed fer = (#correct/#total) frames. The return is
    
    Arguments:
        targets ... a dict with utt_ids with annotated events
        post ... a dict with utt_ids with predicted events
        events ...  maps event text labels to numerical value
    Return:
        updated aligned_scorings with decisions and stats dictionary

    """
    # Checks
    required = [targets,posteriors,events]
    for item in required:
        if not item:
            print(f'Error: {item} is empty.')
            exit(1)
    num_lbl = list(set(events.values()))


    (cor,tot) = 0,0
    for utt_id in targets.keys():
        tgt = io.read_npy(targets[utt_id])
        post = io.read_npy(posteriors[utt_id])
        if tgt.size == post.size:
            print(f'Error {utt_id}: target and post sizes ({tgt.size}/{post.size}) dont match.')
            exit(1)
        if post.shape[1] != len(num_lbl):
            print(f'Error {utt_id}: post shape and events number ({post.shape[1]}/{len(num_lbl)}) dont match.')
            exit(1)            
        pred = post.argmax(1)
        tot += tgt.size
        cor += np.count_nonzero(pred==tgt)
    
    return {'tot':tot, 'cor':cor, 'FER':round(100 - cor/tot*100,2)}