"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import os
from datetime import timedelta
from .parse_scoring import parse_scoring
from sleepat.io import write_scp
from sleepat.utils import list_files, filter_events, string_to_date, date_to_string

def prepare_data(vsn_dir:str, dst_dir:str) -> None:
    """
    Script to prepare VS_10_048 dataset -> to create edf.scp, pariods, annotation,
    utt2spk, spk2utt files. Wave files are bundled inside EDF, which we will extract
    later. This file uses a standard format of ['utt_id'] = {'file': file}, but it
    contains no other informaion (i.e. fs, duration). These values are added when
    we create wave.scp from EDF extracted waveforms. Scoring contains all events from
    only 'good' scorings (marta_scorings). These will be later filtered to find period
    boundaries, which are saved inside periods file, and are used to extract only 
    defined interval from a waveform and to normalize events onsets in the annotation.
    Periods and scorings have a standardized format. Everything is dumped into a dst_dir.
    Input:
        vsn_dir .... folder containg VSN-10-048 dataaset.
        dst_dir .... working directory of the project
    """
    print('Preparing dataset %s' % vsn_dir)

    ## Configuration
    bad_utt = ['VSN-10-048-015']
    marta_scorings = ['ms_snore','ms_snore_v2']
    period_marker = 'analysis-period'
    if not os.path.exists(vsn_dir):
        exit()
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    ## Create the wave.scp file (we extract wave later)
    waves,utt2spk = dict(), dict()
    annot,segments = dict(), dict()
    timestamps = dict()
    print('Creating a list of edf files into %s' % os.path.join(dst_dir,'edf.scp'))
    for file in list_files(vsn_dir,'.edf'):
        utt_id = file.split('.')[0]
        if utt_id in bad_utt:
            continue

        # Wave.scp file (we extract wave later)    
        waves[utt_id] = {'file':os.path.join(vsn_dir,file)}
        utt2spk[utt_id] = utt_id    # Utt_id is a also spk_id, there is 1 file per speaker

        # Annotation file from scoring.json
        events = list()
        json_file = os.path.join(vsn_dir, utt_id+'.scoring.json')
        for scoring in marta_scorings:
            events = events + parse_scoring(json_file,scoring)
        annot[utt_id] = events

        # Segments file, a segment is defined by analysis-period event
        tmp = dict()
        for i, period  in enumerate(filter_events(events,'label',period_marker)):
            seg_id = '-'.join([utt_id,str(i)])
            tmp[seg_id] = {'onset':period['onset'],'duration':period['duration']}
            end = string_to_date(period['start']) + timedelta(seconds=period['duration'])
            timestamps[seg_id] = {'start':period['start'], 'end': date_to_string(end)}
        segments[utt_id] = tmp


    # Dump on disk
    write_scp(os.path.join(dst_dir,'edf.scp'),waves)
    write_scp(os.path.join(dst_dir,'utt2spk'),utt2spk)
    write_scp(os.path.join(dst_dir,'spk2utt'),utt2spk)
    write_scp(os.path.join(dst_dir,'annotation'),annot)
    write_scp(os.path.join(dst_dir,'segments'),segments)
    write_scp(os.path.join(dst_dir,'timestamps'),timestamps)