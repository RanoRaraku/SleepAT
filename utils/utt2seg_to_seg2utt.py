"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
def utt2seg_to_seg2utt(utt2seg:dict) -> dict:
    """
    Transforms mapping inside utt2seg file, which is utt_id->seg_ids,
    to seg_id->utt_id. Utt2seg is one-to-many, seg2utt is many-to-one
    mapping. Used for segmentation scripts.
    Input: utt2seg ...
    Output: seg2utt ... a dictionary
    """
    seg2utt = dict()
    for utt_id, utt2seg in utt2seg.items():
        for seg_id, item  in utt2seg.items():
            #Note: only cause it cant be done on 1 line, maybe create new dict?
            item.update({'utt_id':utt_id})
            seg2utt[seg_id] = item
    return seg2utt
