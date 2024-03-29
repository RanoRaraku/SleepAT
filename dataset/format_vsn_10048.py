"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of local routines specific to VSN-10-048 dataset to
process it into a standard format. All further script assumee an
utterance indexing. The 'utt_id' is THE identifier, it needs to
be unique and the same across all files. The identifier is always
the top-level key for all lists, which are saved as *.scp, and really
are a list of dictionaries or just a list, and are saved as json files.
The processed dataset will contain the following files:
    annotation ... contains all events of interest.
    wave.scp ...contains paths to waveform files.
    utt2spk ... utterance_id to speaker_id mapping
    spk2utt ... speaker_ids to utterance_ids mapping
    utt2seg ...contains info how to segment other files, optional
"""
import os
from os import path
import sleepat
from sleepat import io, utils, opts

def format_vsn_10048(src_dir:str, dst_dir:str, wave_dir:str, config:str=None, **kwargs) -> None:
    """
    Converts prepared data in src_dir to a standardized format. Script extracts specified
    channel from and EDF container and saves them into npy arrays. The rest if mostly copy
    of steps.segment_data().
    Input:
        src_dir ... directory with source files
        dst_dir ... directory to put formatted data_folder
        wave_dir .... where to store waveform extracted from EDF
        <channel> ... channel to extract from EDF container (default:str = 'Audio')
        <valid_events> ... valid events to extract from annotation
            (default:list=['snorebreath',''])
        config
        **kwargs
    """
    print(f'Formatting data folder {src_dir} into {dst_dir}.')
    conf = opts.FormatVSN_10048(config=config, **kwargs)

    if not path.isdir(dst_dir):
        os.mkdir(dst_dir)
    if not path.isdir(wave_dir):
        os.mkdir(wave_dir)


    ## Config section
    waves = io.read_scp(path.join(src_dir,'wave.scp'))
    annot = io.read_scp(path.join(src_dir,'annotation'))
    utt2spk = io.read_scp(path.join(src_dir,'utt2spk'))
    utt2seg = io.read_scp(path.join(src_dir,'utt2seg'))
    segments = io.read_scp(path.join(src_dir,'segments'))


    ## Main part
    waves_new = dict()
    annot_new = dict()
    utt2spk_new = dict()
    for utt_id in utt2spk:
        (wave,fs) = io.read_edf_channel(waves[utt_id]['file'],conf.channel)
        events = utils.filter_scoring(annot[utt_id], 'label', conf.valid_events)

        for (segm_id, segm_wave) in utils.segment_wave(wave,fs,utt2seg[utt_id]):
            file = path.join(wave_dir, segm_id + '.npy')
            waves_new[segm_id] = {'file':file,'fs':fs}
            utt2spk_new[segm_id] = utt_id
            io.write_npy(file, segm_wave)

        for (segm_id,segm_events) in utils.segment_scoring(events,utt2seg[utt_id]):
            annot_new[segm_id] = segm_events

    # Dump on disk
    io.write_scp(path.join(dst_dir,'wave.scp'), waves_new)
    io.write_scp(path.join(dst_dir,'annotation'), annot_new)
    io.write_scp(path.join(dst_dir,'utt2spk'), utt2spk_new)
    io.write_scp(path.join(dst_dir,'spk2utt'), utils.utt2spk_to_spk2utt(utt2spk_new))
    io.write_scp(path.join(dst_dir,'segments'),segments)

    # Final check
    utils.validate_data(dst_dir)
