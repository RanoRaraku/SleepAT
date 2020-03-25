"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
import os
import numpy as np
from sleepat.utils import check_dir, create_segments, segment_events
from sleepat.utils import segment_stamps, segment_wave, utt2spk_to_spk2utt
from sleepat.io import read_scp, write_scp, read_npy, write_npy

def segment_data(data_dir:str, not_wave:bool=True, seg_len:float=10) -> None:
    """
    Segment files the data directory according to the segments file. This means
    splitting/changing files and changing utt_ids to match. Often, We dont want
    to segment waveforms and dump them on disk, but do it on the fly during
    feature extraction, so we just copy wave.scp/segments.
    Input:
        data_dir ... data source directory
        not_wave ... dont segment wave.scp (default:bool = True)
        seg_len ... segment length in seconds (default:float = 10)

    """
    print(f'Segmenting data directory {data_dir}.')

    ## Config and checks
    dst_dir = os.path.join(data_dir,'segmented')
    if not utils.check_dir(dst_dir):
        os.mkdir(dst_dir)
    waves = read_scp(os.path.join(data_dir,'wave.scp'))
    utt2spk = read_scp(os.path.join(data_dir,'utt2spk'))
    annot = read_scp(os.path.join(data_dir,'annotation'))
    stamps = read_scp(os.path.join(data_dir,'timestamps'))
    segments = create_segments(data_dir, seg_len)

    ## Segments annot, utt2spk
    annot_new, utt2spk_new = dict(), dict()
    for utt_id in utt2spk:
        for (seg_id, seg_events) in segment_events(annot[utt_id], segments[utt_id]):
            annot_new[seg_id] = seg_events
            utt2spk_new[seg_id] = utt2spk[utt_id]

    ## Segment timestamps
    stamps_new = dict()
    for utt_id in utt2spk:
        for (seg_id, seg_stamp) in segment_stamps(stamps[utt_id], segments[utt_id]):
            stamps_new[seg_id] = seg_stamp

    ## Handle wave.scp - write info from segments into wave.scp,
    ## and do segmentation later or do it right away.
    waves_new = dict()
    if not_wave:
        write_scp(os.path.join(dst_dir,'wave.scp'),waves)
        write_scp(os.path.join(dst_dir,'segments'),segments)
    else:
        for utt_id in utt2spk:
            fs = waves[utt_id]['fs']
            wave_fid = waves[utt_id]['file']
            wave = read_npy(wave_fid)
            audio_dir = os.path.dirname(os.path.abspath(wave_fid))
            for (seg_id, seg_wave) in segment_wave(wave, fs, segments[utt_id]):
                file = os.path.join(audio_dir, f'{seg_id}.npy')
                write_npy(file, seg_wave, dtype='int16')
                waves_new[seg_id] = {'file':file, 'fs':fs}
        write_scp(os.path.join(dst_dir,'wave.scp'), waves_new)

    ## Dump to disk
    write_scp(os.path.join(dst_dir,'timestamps'), stamps_new)
    write_scp(os.path.join(dst_dir,'annotation'), annot_new)
    write_scp(os.path.join(dst_dir,'utt2spk'), utt2spk_new)
    write_scp(os.path.join(dst_dir,'spk2utt'),
        utils.utt2spk_to_spk2utt(utt2spk_new))
