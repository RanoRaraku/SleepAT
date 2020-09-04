"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import sleepat
from sleepat import io, feat

def feats_to_len(feats_scp:dict) -> tuple:
    """
    Returns feature dimension for each feature file in feats_scp. We assume 0th
    dimesion is number of features and rest is feature dimensions. Number of 
    features is a scalar. The function is a generator.
    Input:
        feats_scp .... scp file containing utt_id and path to file
    Output:
        tuple containing (utterance_id, no. features)
    """
    for utt_id, fid in feats_scp.items():
        feats = io.read_npy(fid)
        yield (utt_id, feats.shape[0])
