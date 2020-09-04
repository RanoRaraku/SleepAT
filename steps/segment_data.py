"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
import os
from os import path
import sleepat
from sleepat import io, utils, opts

def segment_data(data_dir:str, config:str=None, **kwargs) -> None:
    """
    Segment files the data directory according to the utt2seg file. This means
    splitting/changing files and changing utt_ids to match. Often, We dont want
    to segment waveforms and dump them on disk, but do it on the fly during
    feature extraction, so we just copy wave.scp/utt2seg.
    Input:
        data_dir ... data source directory
        <seg_len> ... segment length in seconds (default:float = 10)
        <omit_wave> ... dont segment wave.scp, we do it later (default:bool = True)
        config ... a configuration file that contains optional args. <> (default:str = None)
        **kwargs ... to set optional args. <>
    """
    print(f'Segmenting data directory {data_dir}.')

    ## Config and checks
    conf = opts.SegmentData(config,**kwargs)
    dst_dir = path.join(data_dir,'segmented')
    if not path.exists(dst_dir):
        os.mkdir(dst_dir)
    utils.create_utt2seg(data_dir, conf.seg_len)

    waves = io.read_scp(path.join(data_dir,'wave.scp'))
    utt2spk = io.read_scp(path.join(data_dir,'utt2spk'))
    annot = io.read_scp(path.join(data_dir,'annot'))
    utt2seg = io.read_scp(path.join(data_dir,'utt2seg'))

    ## Segment annot, utt2spk
    annot_new, utt2spk_new = dict(), dict()
    for utt_id in utt2spk.keys():
        for (seg_id, seg_score) in utils.segment_scoring(annot[utt_id],utt2seg[utt_id]):
            annot_new[seg_id] = seg_score
            utt2spk_new[seg_id] = utt2spk[utt_id]

    ## Handle wave.scp - write info from utt2seg into wave.scp,
    waves_new = dict()
    if conf.omit_wave:
        io.write_scp(path.join(dst_dir,'wave.scp'),waves)
        io.write_scp(path.join(dst_dir,'utt2seg'),utt2seg)
    else:
        for utt_id in utt2spk.keys():
            fs = waves[utt_id]['fs']
            wave_fid = waves[utt_id]['file']
            wave = io.read_npy(wave_fid)
            audio_dir = path.dirname(path.abspath(wave_fid))
            for (seg_id, seg_wave) in utils.segment_wave(wave, fs, utt2seg[utt_id]):
                file = path.join(audio_dir, f'{seg_id}.npy')
                io.write_npy(file, seg_wave, dtype='int16')
                waves_new[seg_id] = {'file':file, 'fs':fs}
        io.write_scp(path.join(dst_dir,'wave.scp'), waves_new)

    ## Handle period
    periods_fid = path.join(data_dir,'periods')
    if path.exists(periods_fid):
        periods = io.read_scp(periods_fid)
        periods_new = dict()
        for utt_id in utt2spk.keys():
            for (seg_id, seg_periods) in utils.segment_periods(periods[utt_id], utt2seg[utt_id]):
                    periods_new[seg_id] = seg_periods

    ## Dump to disk
    io.write_scp(path.join(dst_dir,'annot'), annot_new)
    io.write_scp(path.join(dst_dir,'utt2spk'), utt2spk_new)
    io.write_scp(path.join(dst_dir,'periods'), periods_new)
    io.write_scp(path.join(dst_dir,'spk2utt'),
        utils.utt2spk_to_spk2utt(utt2spk_new))

