"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
import os
from os import path
import random
import sleepat
from sleepat import io, utils


def split_data_per_utterance(data_dir:str, dst_dir:str=None,
    subsets:list=['train','dev','eval'], ratio:list=[8,1,1], no_feats:bool=True) -> None:
    """
    Splits the dataset into, by default, train, test, validation subsets with a
    specified ratio.The split is done according to spk2utt. The split is random,
    with indexe drawn from a discreet uniform distribution. The output files go
    into data_dir/subset folders. We assume utt2spk is ordered so we shuffle and
    order again. As of Python 3.7, dicts keep order so shuffling is needed.
    Input:
        data_dir ... directory that contains utt2spk and other files to split
        dst_dir ... output directory (default:str = None)
        subsets ... (default:list = ['train','dev','eval'])
        ratio ... data ratio across subsets (default:list = [8,1,1])
        no_feats ... split products of feature extraction (default:bool = True)
    """
    print(f'Splitting on per-utterance basis {data_dir}.')

    ## Config section
    files_to_split = ['utt2spk','annotation','periods']
    if not no_feats:
        files_to_split.append('feats.scp','targets.scp','mvn.scp')
    utt2spk = io.read_scp(path.join(data_dir,'utt2spk'))
    utt_list = list(utt2spk.keys())
    random.shuffle(utt_list)
    utt_num = len(utt_list)
    if utt_num == 0:
        print('Error: empty utt2spk file.')
        exit()
    if dst_dir is None:
        dst_dir = data_dir

    ## Create corresponding dirs
    for subset in subsets:
        subset_dir = path.join(dst_dir,subset)
        if not path.exists(subset_dir):
            os.makedirs(subset_dir)

    ## Split ids based on utt2spk
    subset_utt = dict()
    idx_i, remainder = 0, 0.0
    for j,subset in enumerate(subsets):
        idx_j = ratio[j]/sum(ratio)*utt_num + idx_i + remainder
        subset_utt[subset] = utt_list[idx_i : round(idx_j)]
        subset_utt[subset].sort()
        remainder = idx_j - round(idx_j)
        idx_i = round(idx_j)

    ## Handle wave.scp/utt2seg
    if path.exists(path.join(data_dir,'utt2seg')):
        wave = io.read_scp(path.join(data_dir,'wave.scp'))
        utt2seg = io.read_scp(path.join(data_dir,'utt2seg'))
        seg2utt = utils.utt2seg_to_seg2utt(utt2seg)
        for subset in subsets:
            subset_seg2utt = {utt: seg2utt[utt] for utt in subset_utt[subset]}
            subset_utt2seg = utils.seg2utt_to_utt2seg(subset_seg2utt)
            subset_wave = {utt: wave[utt] for utt in subset_utt2seg}
            io.write_scp(path.join(dst_dir,subset,'wave.scp'), subset_wave)
            io.write_scp(path.join(dst_dir,subset,'utt2seg'), subset_utt2seg)
    else:
        files_to_split.append('wave.scp')

    ## Split the rest of files
    for file in files_to_split:
        if not path.exists(path.join(data_dir,file)):
            continue
        scp = io.read_scp(path.join(data_dir,file))
        for subset in subsets:
            dst_file = path.join(dst_dir,subset,file)
            subset_scp = {utt: scp[utt] for utt in subset_utt[subset]}
            io.write_scp(dst_file,subset_scp)
            if file == 'utt2spk':
                spk2utt = utils.utt2spk_to_spk2utt(subset_scp)
                io.write_scp(path.join(dst_dir,subset,'spk2utt'),spk2utt)
