"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import os
from os import path
import sleepat
from sleepat import io, utils, opts

def prep_vsn_10048(data_dir:str, dst_dir:str, config:str=None, **kwargs) -> None:
    """
    Script to prepare VSN_10_048 dataset -> to create edf.scp, pariods, annotation,
    utt2spk, spk2utt files. Wave files are bundled inside EDF, which we will extract
    later. This file uses a standard format of ['utt_id'] = {'file': file}, but it
    contains no other informaion (i.e. fs, duration). These values are added when
    we create wave.scp from EDF extracted waveforms. Scoring contains all events from
    only 'good' scorings (marta_scorings). These will be later filtered to find period
    boundaries, which are saved inside periods file, and are used to extract only
    defined interval from a waveform and to normalize events onsets in the annotation.
    Periods and scorings have a standardized format. Everything is dumped into a dst_dir.
    Input:
        data_dir ... folder containg VSN-10-048 dataset.
        dst_dir ... working directory of the project
        <scorings> ... scorings to extract (default:list=['ms_snore','ms_snore_v2'])
        <null_spk> ... speakers to avoid processing (default:list=[''])
        <use_period> ... period to use for segmentation ('analysis'|'recording'),
            (default:str='analysis')
        config ... configuration file to set optional arguments (<>), (default:str=None)
        **kwargs ... keyword arguments to set optiona arguments (<>)
    """
    print(f'Preparing dataset {data_dir}.')
    conf = opts.PrepVSN_10048(config=config, **kwargs)
    if not path.isdir(data_dir):
        exit()
    if not path.isdir(dst_dir):
        os.makedirs(dst_dir)

    ## Create the wave.scp file (we extract wave later)
    wave,utt2spk,segments = dict(), dict(), dict()
    annot,utt2seg = dict(), dict()
    for file in utils.list_files(data_dir,'.edf'):
        utt_id = file.split('.')[0]
        if utt_id in conf.bad_spk:
            continue

        # Wave.scp file (we extract wave later)
        # Utt_id is a also spk_id
        wave[utt_id] = {'file':path.join(data_dir,file)}
        utt2spk[utt_id] = utt_id

        # Annotation file from scoring.json
        events = list()
        scoring_fid = path.join(data_dir, f'{utt_id}.scoring.json')
        data = io.read_scp(scoring_fid)
        for scoring in conf.scorings:
            if scoring not in data:
                print(f'{" ":3}Warning: Scoring "{scoring}" not found for subject {utt_id}.')
                continue
            events += data[scoring]

            if conf.use_period == 'analysis':
                periods = utils.filter_scoring(data[scoring],'label','period_analysis')
                if not periods:
                    msg = (f'{" ":3}Error: No analysis period for scoring "{scoring}"'
                        f' and subject {utt_id}.')
                    print(msg)
                    exit(1)
        annot[utt_id] = events

        tmp = dict()
        if conf.use_period == 'recording':
            periods = [data['Recording']]
        for i,period in enumerate(periods):
            seg_id = '-'.join([utt_id,str(i)])
            segments[seg_id] = period
            tmp[seg_id] = period
        utt2seg[utt_id] = tmp

    # Dump on disk
    io.write_scp(path.join(dst_dir,'wave.scp'),wave)
    io.write_scp(path.join(dst_dir,'utt2spk'),utt2spk)
    io.write_scp(path.join(dst_dir,'spk2utt'),utt2spk)
    io.write_scp(path.join(dst_dir,'annotation'),annot)
    io.write_scp(path.join(dst_dir,'utt2seg'),utt2seg)
    io.write_scp(path.join(dst_dir,'segments'),segments)
