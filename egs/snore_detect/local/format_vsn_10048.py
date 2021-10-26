"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of local routines specific to VSN-10-048 dataset to process
it into a standard format. All further script assumee an recerance
indexing. The 'rec_id' is THE identifier, it needs to be unique and
the same across all files. The identifier is always the top-level key
for all lists, which are saved as *.scp, and really are a list of
dictionaries or just a list, and are saved as json files. The dataset
will contain the following files:
    annot ... contains all events of interest.
    wave.scp ...contains paths to waveform files.
    rec2sub ... recerance_id to sub_id mapping
    sub2rec ... speaker_ids to recerance_ids mapping
    rec2seg ...contains info how to segent other files, optional
"""
import os
from os import path
import sleepat
from sleepat import io, utils, opts

def format_vsn_10048(src_dir:str, dst_dir:str, wave_dir:str, config:str=None, **kwargs) -> None:
    """
    Converts prepared data in src_dir to a standardized format. Script
    extracts specified channel from and HDF5 container and saves them
    into *.npy arrays.
    Input:
        src_dir ... directory with source files
        dst_dir ... directory to put formatted data_folder
        wave_dir .... where to store waveform extracted from EDF
        <channel> ... channel to extract from EDF container (default:str = 'Audio')
        <valid_events> ... valid events to extract from annotation
            (default:list=['snorebreath',''])
        config ... config file to set optional args <>, (default:str=None)
        **kwargs ... to set optional args (<>)
    """
    print(f'Formatting data folder {src_dir} into {dst_dir}.')
    conf = opts.FormatVSN_10048(config=config, **kwargs)

    if not path.isdir(src_dir):
        print(f'Error: {src_dir} not found.')
        exit(1)
    if not path.isdir(dst_dir):
        os.mkdir(dst_dir)
    if not path.isdir(wave_dir):
        os.mkdir(wave_dir)

    waves = io.read_scp(path.join(src_dir,'wave.scp'))
    annot = io.read_scp(path.join(src_dir,'annot'))
    rec2sub = io.read_scp(path.join(src_dir,'rec2sub'))
    rec2seg = io.read_scp(path.join(src_dir,'rec2seg'))
    periods = io.read_scp(path.join(src_dir,'periods'))

    ## Main part
    waves_new = dict()
    annot_new = dict()
    rec2sub_new = dict()
    for rec_id in rec2sub:
        (fs,wave) = io.read_hdf5(waves[rec_id]['file'],conf.channel)
        scoring = utils.filter_scoring(annot[rec_id],'label',conf.valid_events)

        for seg_id,seg_wave in utils.segment_wave(wave,fs,rec2seg[rec_id]):
            file = path.join(wave_dir, f'{seg_id}.{conf.channel}.npy')
            waves_new[seg_id] = {'file':file,'fs': int(fs)}
            rec2sub_new[seg_id] = rec_id
            io.write_npy(file, seg_wave)

        for seg_id, seg_events in utils.segment_scoring(scoring,rec2seg[rec_id]):
            annot_new[seg_id] = seg_events

    # Dump on disk
    io.write_scp(path.join(dst_dir,'wave.scp'),waves_new)
    io.write_scp(path.join(dst_dir,'annot'),annot_new)
    io.write_scp(path.join(dst_dir,'rec2sub'),rec2sub_new)
    io.write_scp(path.join(dst_dir,'sub2rec'),utils.rec2sub_to_sub2rec(rec2sub_new))
    io.write_scp(path.join(dst_dir,'periods'),periods)

    # Final check
    utils.validate_data(dst_dir)
