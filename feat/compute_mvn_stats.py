"""
Made by Michal Borsky, 2019, copyright (C) RU
Acoustuc feature extraction library.
"""
import numpy as np
from sleepat.io import read_npy

def compute_mvn_stats(feats_list:list) -> None :
    """
    Computes mean and variance statistics for a feature or a list
    of features. Uses an combination of batch and online estimation.
    Batch part is estimate stats per feats file and online is update
    statistics over the feats_list. Uses Einstein summation notation
    and power-to-variance relation to speed up variance estimation.
    https://www.johndcook.com/blog/standard_deviation/
    Bishop, "Pattern Recognition and Machine Learning", pp-191.
    Input:
        feats_list ... list of feature files
    Output:
        np.ndarray(shape=(2,feats_num)) that contains [mu,sigma]
    """
    if isinstance(feats_list,str):
        feats_list = [feats_list]

    if len(feats_list) == 0:
        print('feat.compute_mvn_stats(): Feature list is empty.')
        exit()

    total = 0
    for i,file in enumerate(feats_list):
        feats = read_npy(file)
        fnum = feats.shape[0]
        total += fnum
        mu_k = feats.mean(axis=0)
        sigma_k = np.einsum('ij,ij->j',feats,feats)/fnum - mu_k**2
        if i == 0:
            mu = mu_k
            sigma = sigma_k
        else:
            delta_old = (mu_k - mu)
            mu += delta_old*fnum/total
            delta_new = (mu_k - mu)
            sigma += (sigma_k - sigma + delta_old*delta_new)*fnum/total
    return np.array([mu,sigma])
