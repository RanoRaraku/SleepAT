"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
def sub2rec_to_rec2sub(sub2rec:dict) -> dict:
    """
    Convert sub2rec to rec2sub.
    Input: sub2rec.... a dictionary
    Output rec2sub .... a dictionary
    """
    rec2sub = dict()
    for sub_id,rec_ids in sub2rec.items():
        for rec_id in rec_ids:
            rec2sub[rec_id] = sub_id
    return rec2sub
