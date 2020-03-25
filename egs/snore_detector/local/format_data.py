"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of local routines specific to VSN-10-048 dataset to process it into a standard format.
All further script process the dataset on an utterance-basis. The 'utt_id' is THE identifier, it
needs to be unique and the same across all files. The identifier is always the top-level key for
all lists, which are saved as *.scp, and really are a list of dictionaries or just a list, and are
saved as json files. The processed dataset will contain the following files:
    annotation ... contains all events of interest.
    wave.scp ...contains paths to waveform files.
    utt2spk ... utterance_id to speaker_id mapping
    spk2utt ... speaker_ids to utterance_ids mapping
    segments ...contains info how to segment other files, optional.

"""
import os
import numpy as np
from sleepat.io import read_scp, read_edf_channel, write_npy, write_scp
from sleepat.utils import filter_events, segment_wave, segment_events, validate_data
from sleepat.utils import utt2spk_to_spk2utt

def format_data(src_dir:str, dst_dir:str=None, wave_dir:str=None, channel:str = 'Audio'):
    """
    Converts prepared data in src_dir to a standardized format. Script extracts specified
    channel from and EDF container and saves them into npy arrays. The rest if mostly copy
    of steps.segment_data().
    Input:
        src_dir .... directory to find wave.scp, annotation and possibly segments.
        wave_dir .... where to store extracted edf files.
        channel .... which channel to extract from EDF container (default:str = 'Audio')
    """
    print('Formatting data from folder %s into a standard form' % src_dir)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    ## Config section
    waves = read_scp(os.path.join(src_dir,'edf.scp'))
    annot = read_scp(os.path.join(src_dir,'annotation'))
    utt2spk = read_scp(os.path.join(src_dir,'utt2spk'))
    segments = read_scp(os.path.join(src_dir,'segments'))
    stamps = read_scp(os.path.join(src_dir,'timestamps'))
    valid_events = ['snorebreath','breathing-effort']


    ## Main part
    waves_new = dict()
    annot_new = dict()
    utt2spk_new = dict()
    for utt_id in utt2spk:
        print('Extracting channel %s from EDF file %s.' % (channel, waves[utt_id]['file']))
        wave,fs = read_edf_channel(waves[utt_id]['file'],channel)
        events = filter_events(annot[utt_id],'label', valid_events)

        print('Segmenting waveforms.')
        for seg_id,seg_wave in segment_wave(wave,fs, segments[utt_id]):
            file = os.path.join(wave_dir, seg_id + '.npy')
            waves_new[seg_id] = {'file':file,'fs':fs}
            utt2spk_new[seg_id] = utt_id
            write_npy(file, seg_wave, dtype='int16')

        print('Segmenting annotation, onsets will be normalized.')
        for seg_id,seg_events in segment_events(events, segments[utt_id]):
            annot_new[seg_id] = seg_events

    # Dump on disk  
    write_scp(os.path.join(dst_dir,'wave.scp'), waves_new)
    write_scp(os.path.join(dst_dir,'timestamps'), stamps)
    write_scp(os.path.join(dst_dir,'annotation'), annot_new)
    write_scp(os.path.join(dst_dir,'utt2spk'), utt2spk_new)
    write_scp(os.path.join(dst_dir,'spk2utt'), utt2spk_to_spk2utt(utt2spk_new))

    # Final check
    validate_data(dst_dir)
