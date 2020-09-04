"""
Made by Michal Borsky, 2019, copyright (C) RU
Acoustuc feature extraction library.
"""
import os
from os import path
import sleepat 
from sleepat import io, feat, opts



def make_pca_mat(data_dir:str, config:str=None, **kwargs) -> None :
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
    print(f'Computing PCA transform for {data_dir}.')
    conf = opts.ComputePca(config,**kwargs)

    ## Configuration and checks
    feats_scp = io.read_scp(path.join(data_dir,'feats.scp'))
    if len(feats_scp) == 0:
        print('Error: feats.scp is empty.')
        exit()
    if conf.apply_mvn:
        utt2spk = io.read_scp(path.join(data_dir,'utt2spk.scp'))
        mvn_scp = io.read_scp(path.join(data_dir,'mvn.scp'))


    for utt_id, file in feats:
        feats = io.read_npy(file)

        if conf.apply_ma:
            feats = feat.apply_ma(feats,**conf.as_kwargs())
        if conf.apply_mvs:
            feats = feat.apply_mvn(feats,**conf.as_kwargs())
        if conf.apply_mvn:
            spk_id = utt2spk[utt_id]
            feats = feat.apply_mvn(feats, mvn_scp[spk_id],**conf.as_kwargs())
        if conf.add_delta:
            feats = feat.add_delta(feats,**conf.as_kwargs())
        if conf.splice_frames:
            feats = feat.splice_frames(feats,**conf.as_kwargs())
        
        #C = np.cov(feats)
