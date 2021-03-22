"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
import os
from os import path
import sleepat
from sleepat import utils, io, feat


def make_targets(data_dir:str, lang_dir:str, targets_dir:str, config:str=None, **kwargs) -> None:
    """
    Make targets for features extraced using configuration file, where labels
    are changed to numerical values defined in events. It goes over files from
    feats.scp, finds out the feature size, loads a scoring from the annot. file
    and produces a vector for each frame of wstep. The labels in the scoring
    must be defined in the events.

    Arguments:
        data_dir ... data source directory with feats.scp and annotation
        labels_dir ... directory with events
        targets_dir ... target destination directory
        <wstep> ... window step used for segmentation (default:float=0.01)
        config ... config file to load optional arguments (default:str = None)
        **kwargs ... optional arguments
    """
    print(f'Making target files for {data_dir}.')
    for item in [data_dir,lang_dir]:
        if not path.isdir(item):
            print(f'Error: {item} not found.')
            exit(1)
    if not path.isdir(targets_dir):
        os.mkdir(targets_dir)

    utt2spk = io.read_scp(path.join(data_dir,'utt2spk'))
    periods = io.read_scp(path.join(data_dir,'periods'))
    annot = io.read_scp(path.join(data_dir,'annot'))
    events = io.read_scp(path.join(lang_dir,'events'))

    targets = dict()
    for utt_id in utt2spk:
        tgt_fid = path.join(targets_dir, f'{utt_id}.target.npy')
        tgt = utils.scoring_to_targets(annot[utt_id], periods[utt_id], events, config=config, **kwargs)
        io.write_npy(tgt_fid, tgt)
        targets[utt_id] = tgt_fid
    io.write_scp(path.join(data_dir,'targets.scp'), targets)
    print(f'Done.\n')