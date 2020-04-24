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
        <segm_len> ... segment length in seconds (default:float = 10)
        <segm_len_min> ...
        <segm_len_max> ...
        <not_wave> ...
        config ...
        **kwargs ...
    """
    print(f'Segmenting data directory {data_dir}.')

    ## Config and checks
    conf = opts.SegmentDataOpts(config,**kwargs)
    dst_dir = path.join(data_dir,'segmented')
    if not path.exists(dst_dir):
        os.mkdir(dst_dir)

    waves = io.read_scp(path.join(data_dir,'wave.scp'))
    utt2spk = io.read_scp(path.join(data_dir,'utt2spk'))
    annot = io.read_scp(path.join(data_dir,'annotation'))
    utt2seg = utils.create_utt2seg(data_dir, conf.segm_len, conf.segm_len_min)

    ## Segment annot, utt2spk
    annot_new, utt2spk_new = dict(), dict()
    for utt_id in utt2spk.keys():
        for (segm_id, segm_events) in utils.segment_events(annot[utt_id],utt2seg[utt_id]):
            annot_new[segm_id] = segm_events
            utt2spk_new[segm_id] = utt2spk[utt_id]

    ## Handle wave.scp - write info from utt2seg into wave.scp,
    waves_new = dict()
    if conf.not_wave:
        io.write_scp(path.join(dst_dir,'wave.scp'),waves)
        io.write_scp(path.join(dst_dir,'utt2seg'),utt2seg)
    else:
        for utt_id in utt2spk.keys():
            fs = waves[utt_id]['fs']
            wave_fid = waves[utt_id]['file']
            wave = io.read_npy(wave_fid)
            audio_dir = path.dirname(path.abspath(wave_fid))
            for (segm_id, segm_wave) in utils.segment_wave(wave, fs, utt2seg[utt_id]):
                file = path.join(audio_dir, f'{segm_id}.npy')
                io.write_npy(file, segm_wave, dtype='int16')
                waves_new[segm_id] = {'file':file, 'fs':fs}
        io.write_scp(path.join(dst_dir,'wave.scp'), waves_new)


    periods_fid = path.join(data_dir,'periods')
    if path.exists(periods_fid):
        periods = io.read_scp(periods_fid)
        periods_new = dict()
        for utt_id in utt2spk.keys():
            for (segm_id, segm_periods) in utils.segment_periods(periods[utt_id], utt2seg[utt_id]):
                    periods_new[segm_id] = segm_periods

    ## Dump to disk
    io.write_scp(path.join(dst_dir,'annotation'), annot_new)
    io.write_scp(path.join(dst_dir,'utt2spk'), utt2spk_new)
    io.write_scp(path.join(dst_dir,'periods'), periods_new)
    io.write_scp(path.join(dst_dir,'spk2utt'),
        utils.utt2spk_to_spk2utt(utt2spk_new))

