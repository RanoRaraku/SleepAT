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
    Segment files in the data directory according to the rec2seg file. This means
    splitting/changing files and changing rec_ids to match. Often, we dont want
    to segment waveforms and dump them on disk, but do it on the fly during
    feature extraction, so we just copy wave.scp/rec2seg. The files that will be 
    modified are rec2sub, sub2rec, annot, periods.

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
    utils.create_rec2seg(data_dir, conf.seg_len)

    waves = io.read_scp(path.join(data_dir,'wave.scp'))
    rec2sub = io.read_scp(path.join(data_dir,'rec2sub'))
    annot = io.read_scp(path.join(data_dir,'annot'))
    rec2seg = io.read_scp(path.join(data_dir,'rec2seg'))

    ## Segment annot, rec2sub
    annot_new, rec2sub_new = dict(), dict()
    for rec_id in rec2sub.keys():
        for (seg_id, seg_score) in utils.segment_scoring(annot[rec_id],rec2seg[rec_id]):
            annot_new[seg_id] = seg_score
            rec2sub_new[seg_id] = rec2sub[rec_id]

    ## Handle wave.scp - write info from rec2seg into wave.scp,
    waves_new = dict()
    if conf.omit_wave:
        io.write_scp(path.join(dst_dir,'wave.scp'),waves)
        io.write_scp(path.join(dst_dir,'rec2seg'),rec2seg)
    else:
        for rec_id in rec2sub.keys():
            fs = waves[rec_id]['fs']
            wave_fid = waves[rec_id]['file']
            wave = io.read_npy(wave_fid)
            audio_dir = path.dirname(path.abspath(wave_fid))
            for (seg_id, seg_wave) in utils.segment_wave(wave, fs, rec2seg[rec_id]):
                file = path.join(audio_dir, f'{seg_id}.npy')
                io.write_npy(file, seg_wave, dtype='int16')
                waves_new[seg_id] = {'file':file, 'fs':fs}
        io.write_scp(path.join(dst_dir,'wave.scp'), waves_new)

    ## Handle periods files
    periods_fid = path.join(data_dir,'periods')
    if path.exists(periods_fid):
        periods = io.read_scp(periods_fid)
        periods_new = dict()
        for rec_id in rec2sub.keys():
            for (seg_id, seg_periods) in utils.segment_periods(periods[rec_id], rec2seg[rec_id]):
                    periods_new[seg_id] = seg_periods

    ## Dump to disk
    io.write_scp(path.join(dst_dir,'annot'), annot_new)
    io.write_scp(path.join(dst_dir,'rec2sub'), rec2sub_new)
    io.write_scp(path.join(dst_dir,'periods'), periods_new)
    io.write_scp(path.join(dst_dir,'sub2rec'),
        utils.rec2sub_to_sub2rec(rec2sub_new))

