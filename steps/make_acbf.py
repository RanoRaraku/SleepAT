"""
Made by Michal Borsky, 2019, copyright (C) RU
Compute autocorrelation bottleneck features.
"""
import os
from os import path
import sleepat
from sleepat import io, utils, feat

def make_acbf(data_dir:str, feat_dir:str, exp_dir:str, config:str=None, **kwargs) -> None:
    """
    Compute autocorrelation bottleneck features.
    Arguments:
        data_dir .... input data directory
        feat_dir ... output directory for acbf files
        exp_dir ... directory to store autoencoder
        <fs> .... sampling frequency in Hz (default: float = 8000)
        <preemphasis_alpha> ... pre-emphasis coefficient (default: float = 0.97)
        <wlen> ... window length in seconds (default: float = 0.25)
        <wstep> ... window step in seconds (default: float = 0.01)
        <nacf> ... num. of autocorrelation features as input to AutoEnc. (default:int = 320)
        <actype> ... autocorrelation type options include ('same'|'full'),
            (default:str = 'same')
        <nacbf> ... num. of autocorrelation bottleneck features, effectively dimension of
            bottleneck layer (default:int = 16)
        config .... config file to pass optional args. <> (default:str=None)
        **kwargs ... optional args. <>
    """
    print(f'Computing ACBF for {data_dir}.')
    utils.validate_data(data_dir,no_feats=True)

    if not path.isdir(feat_dir):
        os.mkdir(feat_dir)
    wave_scp = io.read_scp(path.join(data_dir,'wave.scp'))
    utt2seg = path.join(data_dir,'utt2seg')
    acf_scp = dict()

    # Compute AutoCorrelation features
    if path.isfile(utt2seg):
        print(f'{" ":3}Utt2seg file found, assuming waveforms are indexed by seg_id.')
        utt2seg = io.read_scp(utt2seg)
        for utt_id, item in wave_scp.items():
            wave = io.read_npy(item['file'])
            for (seg_id, seg_wave) in utils.segment_wave(wave, item['fs'], utt2seg[utt_id]):
                file = path.join(feat_dir, f'{seg_id}.acf.npy')
                acf = feat.compute_acf(seg_wave, config, **kwargs)
                io.write_npy(file, acf)
                acf_scp[seg_id] = file
    else:
        print(f'{" ":3}No utt2seg file found, assuming waveforms are indexed by utt_id.')
        for utt_id, item in wave_scp.items():
            wave = io.read_npy(item['file'])
            file = path.join(feat_dir, f'{utt_id}.acf.npy')
            acf = feat.compute_acf(wave, config, **kwargs)
            io.write_npy(file, acf)
            acf_scp[utt_id] = file
    io.write_scp(path.join(data_dir,'acf.scp'), acf_scp)

    # Train AutoEncoder


    # Extract AutoCorrelation Bottleneck Feature

    print(f'{" ":3}Finished computing ACBF.') 
