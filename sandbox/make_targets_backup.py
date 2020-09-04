"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
import os
from os import path
import sleepat
from sleepat import utils, io, dsp


def make_targets(data_dir:str, lang_dir:str, tgt_dir:str, config:str=None,
    **kwargs) -> None:
    """
    Make targets for features extraced using configuration file, where labels
    are changed to numerical values defined in classes. It goes over files
    from feats.scp, finds out the feature size, loads a corresponding annotation
    and produces a target vector for each frame of wstep. The events labels in
    annotation must be defined in the classes.
    Input:
        data_dir ... data source directory
        lang_dir ... with classes
        tgt_dir ... target destination directory
        <wlen> ... window length used for segmentation (default:float=0.025)
        <wstep> ... window step used for segmentation (default:float=0.01)
        <fs> .... sampling rate in Hz (default:float = 8000.0)
        config ... config file to load optional arguments (default:str = None)
        **kwargs ... optional arguments
    """
    print(f'Making target files for {data_dir}.')
    utils.validate_data(data_dir)
    annot = io.read_scp(path.join(data_dir,'annotation'))
    classes = io.read_scp(path.join(lang_dir,'classes'))
    targets = dict()

    if path.isfile(path.join(data_dir,'segments')):
        print(f'{" ":3}Segments file found, assuming files are indexed by seg_id.')
        utt2seg = io.read_scp(path.join(data_dir,'utt2seg'))
        for segment in utt2seg.values():
            for seg_id, item in segment.items():
                tgt_npy = path.join(tgt_dir, f'{seg_id}.target.npy')
                tgt = utils.annot_to_targets(annot[seg_id], classes,
                    item['duration'], config=config, **kwargs)
                io.write_npy(tgt_npy, tgt)
                targets[seg_id] = tgt_npy
    else:
        print(f'{" ":3}No segments file found, assuming files are indexed by utt_id.')
        wave_scp = io.read_scp(path.join(data_dir,'wave.scp'))
        for utt_id, item in wave_scp.items():
            tgt_npy = path.join(tgt_dir, f'{utt_id}.target.npy')
            no_samples = dsp.wave_to_samples(item['file'])
            tgt = utils.annot_to_targets(annot[utt_id], classes, no_samples,
                config=config, **kwargs)
            io.write_npy(tgt_npy, tgt)
            targets[utt_id] = tgt_npy
    io.write_scp(path.join(data_dir,'targets.scp'), targets)
