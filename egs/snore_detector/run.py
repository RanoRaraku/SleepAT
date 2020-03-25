"""
Script to prepare data from VSN-10-048 dataset for training the snore detector.
Part of dataset was manually scored by Marta Sewartko specifically for this purpose.
The subsets "train_set" and "test_set" are given and core for supervised training.
The remaining subjects can be used for semi-supervised training as we have no realiable
labels for them. Things are stored in numpy arrays and json formats to ensure
compatibility and possibility to use other toolkits other than pyTorch.
"""
import os
import sys
print(sys.path)
import sleepat.steps as steps
import sleepat.utils as utils
import sleepat.egs.snore_detector.local as local


## Config section
stage = 2
vsn_10_048 = '/home/borsky/Projects/Snore/data/VSN-10-048'
work_dir = '/home/borsky/Projects/Snore/snore_detector'
data_dir = os.path.join(work_dir,'data')
feat_dir = os.path.join(work_dir,'feats')
conf_dir = os.path.join(work_dir,'conf')
wave_dir = os.path.join(work_dir,'wave')
bfcc_conf = os.path.join(conf_dir,'bfcc.conf')
lexicon = os.path.join(conf_dir,'lexicon')


## Main
if __name__ == '__main__':

    # Check directories and produce config filesr
    if stage <= 0:
        if not os.path.exists(work_dir):   os.mkdir(work_dir)
        if not os.path.exists(conf_dir):   os.mkdir(conf_dir)
        if not os.path.exists(wave_dir):   os.mkdir(wave_dir)
        if not os.path.exists(data_dir):   os.mkdir(data_dir)
        local.create_lexicon(lexicon)

    # Dataset preparation
    if stage <= 1:
        local.prepare_data(vsn_10_048, f'{data_dir}/tmp')
        local.format_data(f'{data_dir}/tmp', f'{data_dir}/scored', wave_dir, channel='Audio')
        steps.segment_data(f'{data_dir}/scored', not_wave=True)
        steps.split_data_per_utterance(f'{data_dir}/scored/segmented', data_dir, ['train','dev','eval'], [8,1,1])

    # Features extraction, label generation
    if stage <= 2:
        for x in ['train','dev','eval']:
            data_sub = os.path.join(data_dir,x)
            feat_sub = os.path.join(feat_dir,x)
            steps.make_bfcc(data_sub, feat_sub, bfcc_conf)
            utils.merge_scp(f'{data_sub}/bfcc.scp', file_out=f'{data_sub}/feats.scp')
            steps.make_labels(data_sub, lexicon, feat_sub, bfcc_conf)
            steps.make_mvn_stats(data_sub, feat_sub)

    exit()
    # Model Traning - ToDo
    if stage <= 3:
        train_data = os.path.join(data_dir,'train')
        dev_data = os.path.join(data_dir,'dev')
        eval_data = os.path.join(data_dir,'eval')
        exp_dir = os.path.join(work_dir,'exp/cnn')

        steps.train_cnn(train_data, dev_data, exp_dir)
        #steps.decode_cnn(train_data, exp_dir)