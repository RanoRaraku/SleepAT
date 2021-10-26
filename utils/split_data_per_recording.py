"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
import os
from os import path
import random
import sleepat
from sleepat import io, utils


def split_data_per_recording(data_dir:str, dst_dir:str=None,
    subsets:list=['train','dev','eval'], ratio:list=[8,1,1], no_feats:bool=True) -> None:
    """
    Splits the dataset into, by default, train, test, validation subsets with a
    specified ratio. The split is done according to sub2rec. The split is random,
    with indexe drawn from a discreet uniform distribution. The output files go
    into data_dir/subset folders. We assume rec2sub is ordered so we shuffle and
    order again. As of Python 3.7, dicts keep order so shuffling is needed.

    Input:
        data_dir ... directory that contains rec2sub and other files to split
        dst_dir ... output directory (default:str = None)
        <subsets> ... (default:list = ['train','dev','eval'])
        <ratio> ... data ratio across subsets (default:list = [8,1,1])
        <no_feats> ... split products of feature extraction (default:bool = True)
    """
    print(f'Splitting on per-recording basis {data_dir}.')

    ## Config section
    files_to_split = ['rec2sub','annot','periods']
    if not no_feats:
        files_to_split.append('feats.scp','targets.scp','mvn.scp')
    rec2sub = io.read_scp(path.join(data_dir,'rec2sub'))
    rec_list = list(rec2sub.keys())
    random.shuffle(rec_list)
    rec_num = len(rec_list)
    if rec_num == 0:
        print('Error: empty rec2sub file.')
        exit()
    if dst_dir is None:
        dst_dir = data_dir

    ## Create corresponding dirs
    for subset in subsets:
        subset_dir = path.join(dst_dir,subset)
        if not path.exists(subset_dir):
            os.makedirs(subset_dir)

    ## Split ids based on rec2sub
    subset_rec = dict()
    idx_i, remainder = 0, 0.0
    for j,subset in enumerate(subsets):
        idx_j = ratio[j]/sum(ratio)*rec_num + idx_i + remainder
        subset_rec[subset] = rec_list[idx_i : round(idx_j)]
        subset_rec[subset].sort()
        remainder = idx_j - round(idx_j)
        idx_i = round(idx_j)

    ## Handle wave.scp/rec2seg
    if path.exists(path.join(data_dir,'rec2seg')):
        wave = io.read_scp(path.join(data_dir,'wave.scp'))
        rec2seg = io.read_scp(path.join(data_dir,'rec2seg'))
        seg2rec = utils.rec2seg_to_seg2rec(rec2seg)
        for subset in subsets:
            subset_seg2rec = {rec: seg2rec[rec] for rec in subset_rec[subset]}
            subset_rec2seg = utils.seg2rec_to_rec2seg(subset_seg2rec)
            subset_wave = {rec: wave[rec] for rec in subset_rec2seg}
            io.write_scp(path.join(dst_dir,subset,'wave.scp'), subset_wave)
            io.write_scp(path.join(dst_dir,subset,'rec2seg'), subset_rec2seg)
    else:
        files_to_split.append('wave.scp')

    ## Split the rest of files
    for file in files_to_split:
        if not path.exists(path.join(data_dir,file)):
            continue
        scp = io.read_scp(path.join(data_dir,file))
        for subset in subsets:
            dst_file = path.join(dst_dir,subset,file)
            subset_scp = {rec: scp[rec] for rec in subset_rec[subset]}
            io.write_scp(dst_file,subset_scp)
            if file == 'rec2sub':
                sub2rec = utils.rec2sub_to_sub2rec(subset_scp)
                io.write_scp(path.join(dst_dir,subset,'sub2rec'),sub2rec)
