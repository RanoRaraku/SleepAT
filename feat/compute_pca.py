"""
Made by Michal Borsky, 2019, copyright (C) RU
Acoustuc feature extraction library.
"""
import numpy as np

def compute_pca(feats:np.ndarray, pca_dim:int=40) -> None :
    """
    Computes PCA transformation matrix statistics for a feature or a list of features.
    As the PCA is often the last step in feature processing, it expects that input has
    delta-feats, MVN, splicing, and other techniques applied.

    Input:
        datas_dir ... array of values of np.ndarray(shape=(N,M)) shape.
        <pca_dim> ... #pricipal components to retain after transformation (def:int=40)
        <var> ...

    Output:
        np.ndarray(shape=(2,feats_num)) that contains [mu,sigma]
    """
    m,n = feats.shape
    C = np.dot(feats.T,feats) / (m-1)
    eigen_vals, eigen_vecs = np.linalg.eig(C)

    feats_pca = np.dot(feats,eigen_vecs)
