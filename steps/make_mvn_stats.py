"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
import os
from os import path
import sleepat
from sleepat import io, feat

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
    print(f'Computing MVN stats for {data_dir}.')

    ## Configuration and checks
    spk2utt = io.read_scp(path.join(data_dir,'spk2utt'))
    feats_scp = io.read_scp(path.join(data_dir,'feats.scp'))
    mvn_dict = dict()

    for spk, utts in spk2utt.items():
        feats_list = [feats_scp[utt] for utt in utts]
        stats = feat.compute_mvn_stats(feats_list)
        file = path.join(dst_dir,f'{spk}.mvn.npy')
        io.write_npy(file,stats)
        mvn_dict[spk] = file
    io.write_scp(path.join(data_dir,'mvn.scp'), mvn_dict)
    print(f'MVN computation done.')