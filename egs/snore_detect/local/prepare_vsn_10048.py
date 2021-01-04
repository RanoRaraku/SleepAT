"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import os
from os import path
import sleepat
from sleepat import io, utils, opts

def prepare_vsn_10048(data_dir:str, dst_dir:str, config:str=None, **kwargs) -> None:
    """
    Script to prepare VSN_10_048 dataset -> to create edf.scp, periods, annotation,
    utt2spk, spk2utt files. Wave files are bundled inside HDF5, which we will extract
    later. This file uses a standard format of ['utt_id'] = {'file': file}, but it
    contains no other informaion (i.e. fs, duration). These values are added when
    we create wave.scp from HDF5 extracted waveforms. Scoring contains all events from
    only 'good' scorings (marta_scorings). These will be later filtered to find period
    boundaries, which are saved inside periods file, and are used to extract only
    defined interval from a waveform and to normalize events onsets in the annotation.
    Periods and scorings have a standardized format. Everything is dumped into a dst_dir.

    Arguments:
        data_dir ... folder containg VSN-10-048 dataset.
        dst_dir ... working directory of the project
        <scorings> ... scorings to extract (default:list=['ms_snore','ms_snore_v2'])
        <bad_spk> ... speakers to avoid processing (default:list=[''])
        <use_period> ... period to use for segmentation ('analysis'|'recording'|'manual'),
            (default:str='analysis')
        <utt2seg> ... segmentation dict if used_period = 'manual'
        config ... config file to set optional args <>, (default:str=None)
        **kwargs ... to set optional args (<>)
    """
    print(f'Preparing dataset {data_dir} into {dst_dir}.')
    conf = opts.PrepVSN_10048(config=config, **kwargs)

    if not path.isdir(data_dir):
        print(f'Error: {data_dir} not found.')
        exit(1)
    if not path.isdir(dst_dir):
        os.makedirs(dst_dir)

    ## Create the wave.scp file (we extract wave later)
    wave,utt2spk,periods = dict(), dict(), dict()
    annot,utt2seg = dict(), dict()
    for file in utils.list_files(data_dir,'.hdf5'):
        utt_id = file.split('.')[0]
        skip_utt2seg = False
        if utt_id in conf.bad_spk:
            continue

        # Wave.scp file (we extract wave later)
        # Utt_id is a also spk_id
        wave[utt_id] = {'file':path.join(data_dir,file)}
        utt2spk[utt_id] = utt_id

        # Annotation file from scoring.json
        events = list()
        scorings_present = list()
        scoring_fid = path.join(data_dir, f'{utt_id}.scoring.json')
        data = io.read_scp(scoring_fid)
        for scoring in conf.scorings:
            if scoring in data:
                events += data[scoring]
                scorings_present.append(scoring)
            else:
                print(f'Warning: Scoring {scoring} not found for subject {utt_id}.')
        annot[utt_id] = events

        # Periods + utt2seg
        period_lst = list()
        if conf.use_period == 'recording':
            period_lst.append(data['Recording'])
        elif conf.use_period == 'analysis':
            for scoring in scorings_present:
                period = utils.filter_scoring(data[scoring],'label','period_analysis')
                if not period:
                    print(f'Error: no analysis period for {utt_id} and {scoring}.')
                    exit(1)
                period_lst += period   # utils.filter_scoring() returns list so + not append
        elif conf.use_period == 'manual':
            utt2seg[utt_id] = conf.utt2seg[utt_id]
            periods.update(utt2seg[utt_id])
            skip_utt2seg = True                
        else:
            print(f'Error: no valid use_period defined for {utt_id}.')
            exit(1)

        if not skip_utt2seg:
            utt2seg_acc = dict()
            for i,period in enumerate(period_lst):
                seg_id = f'{utt_id}-{str(i)}'
                periods[seg_id] = period
                utt2seg_acc[seg_id] = period
            utt2seg[utt_id] = utt2seg_acc

    # Dump to disk
    io.write_scp(path.join(dst_dir,'wave.scp'),wave)
    io.write_scp(path.join(dst_dir,'utt2spk'),utt2spk)
    io.write_scp(path.join(dst_dir,'spk2utt'),utt2spk)
    io.write_scp(path.join(dst_dir,'annot'),annot)
    io.write_scp(path.join(dst_dir,'utt2seg'),utt2seg)
    io.write_scp(path.join(dst_dir,'periods'),periods)
