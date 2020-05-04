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
    utt2seg ...contains info how to segment other files, optional.
"""
from os import mkdir
from os.path import join, exists
import sleepat.io as io
import sleepat.utils as utils

def format_data(src_dir:str, dst_dir:str=None, wave_dir:str=None, channel:str = 'Audio'):
    """
    Converts prepared data in src_dir to a standardized format. Script extracts specified
    channel from and EDF container and saves them into npy arrays. The rest if mostly copy
    of steps.segment_data().
    Input:
        src_dir .... directory to find wave.scp, annotation and possibly utt2seg.
        wave_dir .... where to store extracted edf files.
        channel .... which channel to extract from EDF container (default:str = 'Audio')
    """
    print('Formatting data from folder %s into a standard form' % src_dir)
    if not exists(dst_dir):
        mkdir(dst_dir)

    ## Config section
    waves = io.read_scp(join(src_dir,'wave.scp'))
    annot = io.read_scp(join(src_dir,'annotation'))
    utt2spk = io.read_scp(join(src_dir,'utt2spk'))
    utt2seg = io.read_scp(join(src_dir,'utt2seg'))
    periods = io.read_scp(join(src_dir,'periods'))
    valid_events = ['snorebreath','breathing-effort']


    ## Main part
    waves_new = dict()
    annot_new = dict()
    utt2spk_new = dict()
    for utt_id in utt2spk:
        print('Extracting channel %s from EDF file %s.' % (channel, waves[utt_id]['file']))
        wave,fs = io.read_edf_channel(waves[utt_id]['file'],channel)
        events = utils.filter_events(annot[utt_id],'label', valid_events)

        print('Segmenting waveforms.')
        for segm_id,segm_wave in utils.segment_wave(wave,fs, utt2seg[utt_id]):
            file = join(wave_dir, segm_id + '.npy')
            waves_new[segm_id] = {'file':file,'fs':fs}
            utt2spk_new[segm_id] = utt_id
            io.write_npy(file, segm_wave, dtype='int16')

        print('Segmenting annotation, onsets will be normalized.')
        for segm_id,segm_events in utils.segment_events(events, utt2seg[utt_id]):
            annot_new[segm_id] = segm_events

    # Dump on disk  
    io.write_scp(join(dst_dir,'wave.scp'), waves_new)
    io.write_scp(join(dst_dir,'annotation'), annot_new)
    io.write_scp(join(dst_dir,'utt2spk'), utt2spk_new)
    io.write_scp(join(dst_dir,'spk2utt'), utils.utt2spk_to_spk2utt(utt2spk_new))
    io.write_scp(join(dst_dir,'periods'),periods)

    # Final check
    utils.validate_data(dst_dir)
