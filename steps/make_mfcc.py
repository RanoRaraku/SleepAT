"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
import os
import numpy as np
from sleepat.utils import validate_data, segment_wave
from sleepat.io import read_scp, write_scp, read_npy, write_npy
from sleepat.feat import compute_mfcc

def make_mfcc(data_dir:str, feat_dir:str, config:str=None, **kwargs) -> None:
    """
    Extract mel-frequency cepstral coefficients. Script assumes existence of wave.scp
    in segments in data_dir.
    Input:
        data_dir .... input directory with wave.scp
        feat_dir .... output directory for mfcc files
        config .... config passed to compute_plp
    """
    print(f'Computing MFCC features for {data_dir}.')

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
                file = os.path.join(feat_dir, f'{seg_id}.mfcc.npy')
                mfcc = compute_mfcc(seg_wave, config, **kwargs)
                write_npy(file, mfcc)
                feats_dict[seg_id] = file
    write_scp(os.path.join(data_dir,'mfcc.scp'), feats_dict)
