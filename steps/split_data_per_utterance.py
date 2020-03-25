"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
from os import makedirs
from os.path import isdir, isfile, join
from random import shuffle
from sleepat.utils import segments_to_seg2utt, seg2utt_to_segments
from sleepat.utils import utt2spk_to_spk2utt
from sleepat.io import read_scp, write_scp

def split_data_per_utterance(data_dir:str, dst_dir:str=None,
    subsets:list=['train','dev','eval'], ratio:list=[8,1,1]) -> None:
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
        ratio ... (default:list = [8,1,1])
    """
    print(f'Splitting on per-utterance basis {data_dir}.')

    ## Config section
    files_to_split = ['utt2spk','annotation','feats.scp','targets.scp', 'timestamps']
    utt2spk = read_scp(join(data_dir,'utt2spk'))
    utt_list = list(utt2spk.keys())
    shuffle(utt_list)
    utt_num = len(utt_list)
    if utt_num == 0:
        print('Error: empty utt2spk file.')
        exit()
    if dst_dir is None:
        dst_dir = data_dir

    ## Create corresponding dirs
    for subset in subsets:
        if not join(join(dst_dir,subset)):
            makedirs(join(dst_dir,subset))

    ## Main- split ids based on utt2spk
    subset_utt = dict()
    idx_i, remainder = 0, 0.0
    for j,subset in enumerate(subsets):
        idx_j = ratio[j]/sum(ratio)*utt_num + idx_i + remainder
        subset_utt[subset] = utt_list[idx_i : round(idx_j)]
        subset_utt[subset].sort()
        remainder = idx_j - round(idx_j)
        idx_i = round(idx_j)

    ## Handle wave.scp/segments
    if join(join(data_dir,'segments')):
        wave = read_scp(join(data_dir,'wave.scp'))
        segments = read_scp(join(data_dir,'segments'))
        seg2utt = segments_to_seg2utt(segments)
        for subset in subsets:
            subset_seg2utt = {utt: seg2utt[utt] for utt in subset_utt[subset]}
            subset_segments = seg2utt_to_segments(subset_seg2utt)
            subset_wave = {utt: wave[utt] for utt in subset_segments}
            write_scp(join(dst_dir,subset,'wave.scp'), subset_wave)
            write_scp(join(dst_dir,subset,'segments'), subset_segments)
    else:
        files_to_split.append('wave.scp')

    ## Split the rest of files
    for file in files_to_split:
        if not join(join(data_dir,file)):
            continue
        scp = read_scp(join(data_dir,file))
        for subset in subsets:
            dst_file = join(dst_dir,subset,file)
            subset_scp = {utt: scp[utt] for utt in subset_utt[subset]}
            write_scp(dst_file,subset_scp)
            if file == 'utt2spk':
                spk2utt = utt2spk_to_spk2utt(subset_scp)
                write_scp(join(dst_dir,subset,'spk2utt'),spk2utt)
