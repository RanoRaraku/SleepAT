"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
def spk2utt_to_utt2spk(spk2utt:dict) -> dict:
    """
    Convert spk2utt to utt2spk.
    Input: spk2utt.... a dictionary
    Output utt2spk .... a dictionary
    """
    utt2spk = dict()
    for spk_id,utt_ids in spk2utt.items():
        for utt_id in utt_ids:
            utt2spk[utt_id] = spk_id
    return utt2spk
