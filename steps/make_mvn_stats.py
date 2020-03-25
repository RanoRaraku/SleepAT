"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
import os
import numpy as np
from sleepat.io import read_scp, write_scp, write_npy
from sleepat.feat import compute_mvn_stats

def make_mvn_stats(data_dir:str, dst_dir:str) -> None:
    """
    Make statistics for mean and variance normalization. The default
    setup is per-speaker, but its possible to do per-utterance. The
    mvn.npy go into same folder where feats files are. The mvn.scp
    goes into data folder.
    Input:
        data_dir ... input directory with spk2utt and feats.scp
        dst_dir ... output directory for npy stats files
    """
    print(f'Computing cmvn stats for {data_dir}.')

    ## Configuration and checks
    spk2utt = read_scp(os.path.join(data_dir,'spk2utt'))
    feats_dict = read_scp(os.path.join(data_dir,'feats.scp'))
    mvn_dict = dict()

    for spk, utts in spk2utt.items():
        feats_list = [feats_dict[utt] for utt in utts]
        stats = compute_mvn_stats(feats_list)
        file = os.path.join(dst_dir,f'{spk}.mvn.npy')
        write_npy(file,stats)
        mvn_dict[spk] = file
    write_scp(os.path.join(data_dir,'mvn.scp'), mvn_dict)
