"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
def seg2rec_to_rec2seg(seg2rec:dict) -> dict:
    """
    Transforms mapping from seg_id->rec_id to rec_id->seg_ids,
    which is a standard format for rec2seg file. Seg2rec is
    many-to-one, rec2seg is one-to-many mapping. Used for segmentation scripts.
    
    Input: seg2rec ...
    Output: rec2seg ...
    """
    rec2seg = dict()
    for seg_id, item in seg2rec.items():
        cur_rec = item['rec_id']
        if not cur_rec in rec2seg:
            rec2seg[cur_rec] = dict()
        tmp = {seg_id : {'onset':item['onset'],'duration':item['duration']}}
        rec2seg[cur_rec].update(tmp)
    return rec2seg
