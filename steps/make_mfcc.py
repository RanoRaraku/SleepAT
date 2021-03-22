"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
import os
from os import path
import sleepat
from sleepat import io, utils, feat

def make_mfcc(data_dir:str, feat_dir:str, config:str=None, **kwargs) -> None:
    """
    Extract mel-frequency cepstral coefficients. Script assumes existence
    of wave.scp in data_dir.
    Input:
        data_dir .... input data directory
        feat_dir .... output directory for mfcc files
        <fs> .... sampling frequency in Hz (default: float = 8000)
        <preemphasis_alpha> ... pre-emphasis coefficient (default: float = 0.97)
        <wlen> ... window length in seconds (default: float = 0.25)
        <wstep> ... window step in seconds (default: float = 0.01)
        <mel_filts>  .... number of filters (default: int = 22)
        <fmin> ... minimal frequency (default: float = 0)
        <fmax> ... maximum frequency (default: float = fs/2)
        <nceps> ... num. of cepstral coefficients including 0th (default:int = 13)
        config .... config file to pass optional args. <> (default:str=None)
        **kwargs ... optional args. <>
    """
    print(f'Computing MFCC features for {data_dir}.')
    utils.validate_data(data_dir,no_feats=True)

    if not path.isdir(feat_dir):
        os.mkdir(feat_dir)
    wave_scp = io.read_scp(path.join(data_dir,'wave.scp'))
    utt2seg_scp = path.join(data_dir,'utt2seg')
    feats_dict = dict()

    # Main part
    if path.isfile(utt2seg_scp):
        print(f'Utt2seg file found, assuming waveforms are indexed by seg_id.')
        utt2seg = io.read_scp(utt2seg_scp)
        for utt_id, item in wave_scp.items():
            wave = io.read_npy(item['file'])
            for (seg_id, seg_wave) in utils.segment_wave(wave, item['fs'], utt2seg[utt_id]):
                file = path.join(feat_dir, f'{seg_id}.mfcc.npy')
                mfcc = feat.compute_mfcc(seg_wave, config, **kwargs)
                io.write_npy(file, mfcc)
                feats_dict[seg_id] = file
    else:
        print(f'No utt2seg file found, assuming waveforms are indexed by utt_id.')
        for utt_id, item in wave_scp.items():
            wave = io.read_npy(item['file'])
            file = path.join(feat_dir, f'{utt_id}.mfcc.npy')
            mfcc = feat.compute_mfcc(wave, config, **kwargs)
            io.write_npy(file, mfcc)
            feats_dict[utt_id] = file
    io.write_scp(path.join(data_dir,'mfcc.scp'), feats_dict)
    
    print(f'MFCC extraction done.') 
