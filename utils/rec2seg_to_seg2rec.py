"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
def rec2seg_to_seg2rec(rec2seg:dict) -> dict:
    """
    Transforms mapping inside rec2seg file, which is rec_id->seg_ids,
    to seg_id->rec_id. rec2seg is one-to-many, seg2rec is many-to-one
    mapping. Used for segmentation scripts.

    Input: rec2seg ... a dict that maps from recording_ids to segment_ids
    Output: seg2rec ... a dict that maps from segment_ids to recording_ids
    """
    seg2rec = dict()
    for rec_id, rec2seg in rec2seg.items():
        for seg_id, item  in rec2seg.items():
            
            #Note: only cause it cant be done on 1 line, maybe create new dict?
            item.update({'rec_id':rec_id})
            seg2rec[seg_id] = item
    return seg2rec
