"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
import os
import numpy as np
from sleepat.utils import validate_data, segment_wave
from sleepat.io import read_scp, write_scp, read_npy, write_npy
from sleepat.feat import compute_fbank

def make_fbank(data_dir:str, feat_dir:str, config:str=None, **kwargs) -> None:
    """
    Extract fbank features. Script assumes existence of wave.scp, mfcc.conf and optionally
    segments in correct location to function.
    Input:
        data_dir .... input directory with wave.scp
        feat_dir .... output directory for mfcc files
        fbank_conf .... config passed to compute_mfcc (default:str = conf/mfcc.conf)
    """
    print(f'Computing fbank features for {data_dir}.')
    validate_data_dir(data_dir)

    if not os.path.isdir(feat_dir):
        print(f'Creating feature directory {feat_dir}.')
        os.mkdir(feat_dir)
    wave_dict = read_scp(os.path.join(data_dir,'wave.scp'))
    feats_dict = dict()

    # Main part
    if os.path.isfile(os.path.join(data_dir,'segments')):
        print('Segments file found, using that.')

        seg_dict = read_scp(os.path.join(data_dir,'segments'))
        for utt_id, item in wave_dict.items():
            wave = read_npy(item['file'])
            for (seg_id, seg_wave) in segment_wave(wave, item['fs'], seg_dict[utt_id]):
                file = os.path.join(feat_dir, f'{seg_id}.fbank.npy')
                fbank = compute_fbank(seg_wave, config, **kwargs)
                write_npy(file, fbank)
                feats_dict[seg_id] = file
    write_scp(os.path.join(data_dir,'fbank.scp'), feats_dict)
