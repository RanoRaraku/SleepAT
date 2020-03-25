"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
import os
import numpy as np
from sleepat.utils import validate_data, feats_to_len
from sleepat.io import read_scp, write_scp, write_npy
from sleepat.encode import nominal


def make_targets(data_dir:str, lexicon:str, dst_dir:str=None, config:str=None, **kwargs) -> None:
    """
    Make targets for features extraced using configuration file, where labels are changed
    to numerical values defined in lexicon. It goes over files from feats.scp, finds out
    the feature size, loads a corresponding annotation and produces a target vector for
    each frame of wstep. The events labels in annotation must be defined in the lexicon.
    Input:
        data_dir ... data source directory
        dst_dir .... destination directory for targets
        lexicon .... maps event labels to ordinal numbers
        <wstep> .... (default:float = 0.01)
        config
        **kwargs .... allow to set wstep from a config file
    TODO:
        allow feat2dim to exist which would be loaded instead of feats_scp
        You use wstep=0.01 here!
    """
    print(f'Making target files for {data_dir}.')

    ## Config section
    validate_data(data_dir)
    if not os.path.isfile(lexicon):
        print(f'Error: Could not find lexicon file: {lexicon}.')
        exit()
    feats = read_scp(os.path.join(data_dir,'feats.scp'))
    annot = read_scp(os.path.join(data_dir,'annotation'))
    lexicon_dict = read_scp(lexicon)
    targets = dict()

    ## Main
    for utt_id, tlen in feats_to_len(feats):
        events = annot[utt_id]
        if dst_dir is None:
            dst_dir = os.path.dirname(os.path.abspath(feats[utt_id]))
        target_file = os.path.join(dst_dir, f'{utt_id}.target.npy')
        target = nominal(events, lexicon_dict, tlen, config, **kwargs)
        write_npy(target_file, target)
        targets[utt_id] = target_file
    write_scp(os.path.join(data_dir,'targets.scp'), targets)
