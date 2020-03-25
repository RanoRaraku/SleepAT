"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""

def segments_to_seg2utt(segments:dict) -> dict:
    """
    Transforms mapping inside segments file, which is utt_id->seg_ids,
    to seg_id->utt_id. Utt2seg is one-to-many, seg2utt is many-to-one
    mapping. Used for segmentation scripts.
    Input: segments ...
    Output: seg2utt ... a dictionary
    """
    seg2utt = dict()
    for utt_id, segments in segments.items():
        for seg_id, item  in segments.items():
            #Note: only cause it cant be done on 1 line, maybe create new dict?
            item.update({'utt_id':utt_id})
            seg2utt[seg_id] = item
    return seg2utt
