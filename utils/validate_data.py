"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
import os
from sleepat.io import read_scp
from sleepat.utils import spk2utt_to_utt2spk, get_nested_dict_items

def validate_data(data_dir:str) -> None:
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
    if not os.path.isdir(data_dir):
        exit()

    required = ['utt2spk','spk2utt','wave.scp','annotation']
    if os.path.isfile(os.path.join(data_dir,'feats.scp')):
        required.append('feats.scp')
    if os.path.isfile(os.path.join(data_dir,'targets.scp')):
        required.append('targets.scp')

    for file in required:
        if not os.path.isfile(os.path.join(data_dir,file)):
            print('Error: file %s is missing' % file)
            exit()

    ## Get utt_ids from utt2spk
    utt_ids = sorted(read_scp(os.path.join(data_dir,required[0])).keys())

    ## Check utt2spk and spk2utt
    spk2utt = read_scp(os.path.join(data_dir,'spk2utt'))
    if not utt_ids == sorted(spk2utt_to_utt2spk(spk2utt).keys()):
            print('Error: Utterance ids in spk2utt dont match with utt2spk.')
            exit()

    ## Check utt_ids for wave.scp
    wave_dict = read_scp(os.path.join(data_dir,'wave.scp'))
    if os.path.isfile(os.path.join(data_dir,'segments')):
        seg_dict = read_scp(os.path.join(data_dir,'segments'))
        if not sorted(wave_dict.keys()) == sorted(seg_dict.keys()):
            print('Error: Utterance ids in wave.scp dont match with segments.')
            exit()
        wave_ids = [key for (key,_) in get_nested_dict_items(seg_dict, depth = 1)]
    else:
        wave_ids = wave_dict.keys()
    if not utt_ids == sorted(wave_ids):
            print('Error: Utterance ids in utt2spk dont match with wave.scp.')
            exit()

    ## Check utt_ids for other files
    for file in required[3:]:
        file_dict = read_scp(os.path.join(data_dir,file))
        if not utt_ids == sorted(file_dict.keys()):
            print('Error: Utterance ids in %s dont match with %s.' % (file,required[0]))
            exit()
    print('Successfully validated the directory.')
