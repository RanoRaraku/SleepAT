"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
from os.path import join, abspath, dirname, isfile
from sleepat.utils import validate_data, annot_to_targets, wave_to_len
from sleepat.io import read_scp, write_scp, write_npy


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
        <wlen> ....
        <wstep> ....
        config .... 
        **kwargs .... allow to set wstep from a config file
    """
    print(f'Making target files for {data_dir}.')
    validate_data(data_dir)
    annot = read_scp(join(data_dir,'annotation'))
    classes = read_scp(join(lang_dir,'classes'))
    targets = dict()

    if isfile(join(data_dir,'segments')):
        print('Segments file found, assuming files are indexed by segments.')
        segm_scp = read_scp(join(data_dir,'segments'))
        for segments in segm_scp.values():
            for segm_id, item in segments.items():
                tgt_npy = join(tgt_dir, f'{segm_id}.target.npy')
                dur = item['duration']
                tgt = annot_to_targets(annot[segm_id], dur, classes,
                    config=config, **kwargs)
                write_npy(tgt_npy, tgt)
                targets[segm_id] = tgt_npy
    else:
        print('No segments file found, assuming files are indexed by utterance.')        
        wave_scp = read_scp(join(data_dir,'wave.scp'))
        for utt_id, item in wave_scp.items():
            tgt_npy = join(tgt_dir, f'{utt_id}.target.npy')
            dur = wave_to_len(item['file'],item['fs'])
            tgt = annot_to_targets(annot[utt_id], dur, classes,
                config=config, **kwargs)                
            write_npy(tgt_npy, tgt)
            targets[utt_id] = tgt_npy
    write_scp(join(data_dir,'targets.scp'), targets)