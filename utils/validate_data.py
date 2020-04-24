"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import os
from os import path
import sleepat
from sleepat import io, utils

def validate_data(data_dir:str, no_feats:bool = True) -> None:
    """
    Checks the data directory. Invoked after data preparation. The objective is to
    make sure the precessed dataset is in a standardized format. The scp files are 
    dictionaries so they dont need to be sorted, but they need to have identical
    identifiers across all files as top level keys. The identifier is utt_id.
    Input:
        data_dir .... directory to check
        no_feats .... dont check feats.scp (default:bool = False)
        no_targets .... dont check targets.scp (default:bool = False)
    """
    print('Validating data directory %s.' % data_dir)
    if not path.isdir(data_dir):
        exit()

    required = ['utt2spk','spk2utt','wave.scp','annotation']
    if not no_feats:
        required += ['feats.scp','targets.scp']

    for file in required:
        if not path.isfile(path.join(data_dir,file)):
            print('Error: file %s is missing' % file)
            exit()

    ## Get utt_ids from utt2spk
    utt_ids = sorted(io.read_scp(path.join(data_dir,required[0])).keys())

    ## Check utt2spk and spk2utt
    spk2utt = io.read_scp(path.join(data_dir,'spk2utt'))
    if not utt_ids == sorted(utils.spk2utt_to_utt2spk(spk2utt).keys()):
            print('Error: Utterance ids in spk2utt dont match with utt2spk.')
            exit()

    ## Check utt_ids for wave.scp
    wave_dict = io.read_scp(path.join(data_dir,'wave.scp'))
    if path.isfile(path.join(data_dir,'utt2seg')):
        utt2seg = io.read_scp(path.join(data_dir,'utt2seg'))
        if not sorted(wave_dict.keys()) == sorted(utt2seg.keys()):
            print('Error: Utterance ids in wave.scp dont match with utt2seg.')
            exit()
        wave_ids = [key for (key,_) in utils.get_nested_dict_items(utt2seg, depth = 1)]
    else:
        wave_ids = wave_dict.keys()
    if not utt_ids == sorted(wave_ids):
            print('Error: Utterance ids in utt2spk dont match with wave.scp.')
            exit()

    ## Check utt_ids for other files
    for file in required[3:]:
        file_dict = io.read_scp(path.join(data_dir,file))
        if not utt_ids == sorted(file_dict.keys()):
            print('Error: Utterance ids in %s dont match with %s.' % (file,required[0]))
            exit()
    print('Successfully validated the directory.')
