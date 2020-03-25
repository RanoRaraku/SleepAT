"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
from sleepat.io import read_npy

def feats_to_len(feats_scp) -> int:
    """
    Returns feature dimension for each feature file in feats_scp.
    We assume 0th dimension is feature length and other dimensions
    are feature dimensions.
    Input:
        feats_scp .... either an scp or a path to scp
    Output:
        (utt_id, feat_len) tuple
    """
    if isinstance(feats_scp,dict):
        pass
    else:
        print('utils.feat_to_fim(): Wrong input type, expected dict.')
        exit(1)

    for utt_id, fid in feats_scp.items():
        feats = read_npy(fid)
        yield (utt_id, feats.shape[0])
