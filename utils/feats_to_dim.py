"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import sleepat
from sleepat import io

def feats_to_dim(feats_scp:dict) -> tuple:
    """
    Returns feature dimension for each feature file in feats_scp.
    We assume 0th dimesion is number of features and rest is feature
    dimensions. Feature dimensions is a tuple to account for tensor
    features.

    Arguments:
        feats_scp .... scp file contianing utt_id and path to file
         ... (default:bool = False)
        tuple containing (utterance_id, feature dimensions)
    """
    for utt_id, fid in feats_scp.items():
        feats = io.read_npy(fid)
        yield (utt_id, feats.shape[1:])