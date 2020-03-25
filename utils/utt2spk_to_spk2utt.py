"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""

def utt2spk_to_spk2utt(utt2spk:dict) -> dict:
    """
    Convert utt2spk to spk2utt. Uses converstion ->set
    to get a unique list of speaker ids and pre-allocation
    of spk2utt. The value for each spk_id is a list of utt_ids.
    Input: utt2spk .... a dictionary
    Output spk2utt .... a dictionary
    """
    # Note: is this necessary, I got over it twice
    spk2utt = dict()
    for utt_id, spk_id in utt2spk.items():
        if not spk_id in spk2utt:
            spk2utt[spk_id] = list()
        spk2utt[spk_id].append(utt_id)
    return spk2utt