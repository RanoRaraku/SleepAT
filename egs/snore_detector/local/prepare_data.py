"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
from os import makedirs
from os.path import exists, join
import sleepat.io as io
import sleepat.utils as utils
import sleepat.egs.snore_detector.local as local

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
    null_spk = ['VSN-10-048-015']
    marta_scorings = ['ms_snore','ms_snore_v2']
    period_marker = 'analysis-period'
    if not exists(vsn_dir):
        exit()
    if not exists(dst_dir):
        makedirs(dst_dir)

    ## Create the wave.scp file (we extract wave later)
    wave,utt2spk,periods = dict(), dict(), dict()
    annot,utt2seg = dict(), dict()
    print('Creating a list of edf files into %s' % join(dst_dir,'edf.scp'))
    for file in utils.list_files(vsn_dir,'.edf'):
        utt_id = file.split('.')[0]
        if utt_id in null_spk:
            pass

        # Wave.scp file (we extract wave later)    
        wave[utt_id] = {'file':join(vsn_dir,file)}
        utt2spk[utt_id] = utt_id    # Utt_id is a also spk_id, there is 1 file per speaker

        # Annotation file from scoring.json
        events = list()
        json_file = join(vsn_dir, utt_id+'.scoring.json')
        for scoring in marta_scorings:
            events = events + local.parse_scoring(json_file,scoring)
        annot[utt_id] = events

        # utt2seg file, a segment is defined by the "analysis-period" event
        tmp = dict()
        for i, period  in enumerate(utils.filter_events(events,'label',period_marker)):
            seg_id = '-'.join([utt_id,str(i)])
            tmp[seg_id] = {'start':period['start'],'onset':period['onset'],'duration':period['duration']}
            periods[seg_id] = {'start':period['start'],'duration':period['duration']}
        utt2seg[utt_id] = tmp


    # Dump on disk
    io.write_scp(join(dst_dir,'wave.scp'),wave)
    io.write_scp(join(dst_dir,'utt2spk'),utt2spk)
    io.write_scp(join(dst_dir,'spk2utt'),utt2spk)
    io.write_scp(join(dst_dir,'annotation'),annot)
    io.write_scp(join(dst_dir,'utt2seg'),utt2seg)
    io.write_scp(join(dst_dir,'periods'),periods)