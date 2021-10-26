"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
def rec2sub_to_sub2rec(rec2sub:dict) -> dict:
    """
    Convert rec2sub to sub2rec. Uses converstion ->set
    to get a unique list of speaker ids and pre-allocation
    of sub2rec. The value for each sub_id is a list of rec_ids.
    Input: rec2sub .... a dictionary
    Output sub2rec .... a dictionary
    """
    # Note: is this necessary, I got over it twice
    sub2rec = dict()
    for rec_id, sub_id in rec2sub.items():
        if not sub_id in sub2rec:
            sub2rec[sub_id] = list()
        sub2rec[sub_id].append(rec_id)
    return sub2rec
