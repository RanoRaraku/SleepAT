"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
def seg2utt_to_utt2seg(seg2utt:dict) -> dict:
    """
    Transforms mapping from seg_id->utt_id to utt_id->seg_ids,
    which is a standard format for utt2seg file. Seg2utt is
    many-to-one, utt2seg is one-to-many mapping. Used for
    segmentation scripts.
    Input: seg2utt ...
    Output: utt2seg ...
    """
    utt2seg = dict()
    for seg_id, item in seg2utt.items():
        cur_utt = item['utt_id']
        if not cur_utt in utt2seg:
            utt2seg[cur_utt] = dict()
        tmp = {seg_id : {'onset':item['onset'],'duration':item['duration']}}
        utt2seg[cur_utt].update(tmp)
    return utt2seg
