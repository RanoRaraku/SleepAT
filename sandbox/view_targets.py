"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.

Not ready for use!
"""
import os
from os import path
import numpy as np
import sleepat
from sleepat import io, feat,dsp
import matplotlib
from matplotlib import pyplot


def view_targets(data_dir:str, conf:str) -> None:
    """
    Visualized targets and features alignment. Used purely to manually
    check for errors. Expects targets.scp and feats.scp is present in
    data_dir. It will loop over all feats files and plots features and
    targets.
    Input:
        data_dir ...
        mode ...

    """
    conf = io.read_scp(conf)
    wave_fid = path.join(data_dir,'feats.scp')
    targets_fid = path.join(data_dir,'targets.scp')
    mvn_fid = path.join(data_dir,'mvn.scp')
    utt2spk_fid = path.join(data_dir,'utt2spk')

    wave_scp = io.read_scp(wave_fid)
    targets_scp = io.read_scp(targets_fid)
    if path.isfile(mvn_fid):
        mvn_scp = io.read_scp(mvn_fid)
        utt2spk_scp = io.read_scp(utt2spk_fid)

    for utt_id, wav_npy in wave_scp.items():
        wave = io.read_npy(wav_npy)
        wave = wave-wave.mean()
        wave = wave/wave.std()
        targets = io.read_npy(targets_scp[utt_id])
        #targets = targets.repeat(conf['wstep']*conf['fs'])

        pyplot.plot(wave)
        pyplot.plot(2*targets)
        pyplot.title(utt_id)
        pyplot.show()

    """
    for utt_id, feats_npy in feats_scp.items():
        feats = io.read_npy(feats_npy)
        targets = io.read_npy(targets_scp[utt_id])
        if mvn_scp :
            spk_id = utt2spk_scp[utt_id]
            feats = feat.apply_mvn(feats,mvn_scp[spk_id], norm_vars = True)

        pyplot.plot(feats)
        pyplot.plot(2*targets)
        pyplot.show()
    """