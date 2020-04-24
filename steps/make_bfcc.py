"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
import os
from os import path
import sleepat
from sleepat import io, utils, feat

def make_bfcc(data_dir:str, feat_dir:str, config:str=None, **kwargs) -> None:
    """
    Extract bark-frequency cepstral coefficients. Script assumes existence of wave.scp
    in segments in data_dir.
    Input:
        data_dir .... input directory with wave.scp
        feat_dir .... output directory for bfcc files
        config .... config passed compute_bfcc()
        **kwargs .... keyworgs args. passed to compute_bfcc()
    """
    print(f'Computing BFCC features for {data_dir}.')
    utils.validate_data(data_dir,no_feats=True)

    if not path.isdir(feat_dir):
        print(f'Creating feature directory {feat_dir}.')
        os.mkdir(feat_dir)
    wave_scp = io.read_scp(path.join(data_dir,'wave.scp'))
    segm_scp = path.join(data_dir,'segments') 
    feats_dict = dict()

    # Main part
    if path.isfile(segm_scp):
        print('Segments file found, using it to extract features.')
        segm_dict = io.read_scp(segm_scp)
        for utt_id, item in wave_scp.items():
            wave = io.read_npy(item['file'])
            for (segm_id, segm_wave) in utils.segment_wave(wave, item['fs'], segm_dict[utt_id]):
                file = path.join(feat_dir, f'{segm_id}.bfcc.npy')
                bfcc = feat.compute_bfcc(segm_wave, config, **kwargs)
                io.write_npy(file, bfcc)
                feats_dict[segm_id] = file
    io.write_scp(path.join(data_dir,'bfcc.scp'), feats_dict)
