"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
from sleepat.io import read_npy

def feats_to_dim(feats_scp) -> tuple:
    """
    Returns feature dimension for each feature file in
    feats_scp. We assume all feats have same size, so we
    use return to break loop and return dimension of 1st file.
    Also, we assume 0th dimesion is feature length.
    Input:
        feats_scp .... either an scp or a path to scp
    Output:
        feat_dim tuple
    """
    if isinstance(feats_scp,dict):
        pass
    else:
        print('utils.feat_to_fim(): Wrong input type, expected dict.')
        exit(1)

    for fid in feats_scp.values():
        feats = read_npy(fid)
        return feats.shape[1:]
