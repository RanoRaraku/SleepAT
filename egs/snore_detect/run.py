"""
Michal Borsky, Reykjavik University, 2019.

Main script to train a snore detector. Needs to be copied,
not only linked to work properly.
"""
import os
from os import path
import sleepat
from sleepat import steps, utils
from sleepat.egs.snore_detector import local


## Config section
stage = 4
vsn_10_048 = '/home/borsky/datasets/VSN-10-048-MS_scored/EDF'
work_dir = '/home/borsky/Projects/snore_detection'
data_dir = path.join(work_dir,'data')
feat_dir = path.join(work_dir,'feats')
conf_dir = path.join(work_dir,'conf')
wave_dir = path.join(work_dir,'wave')
exp_dir = path.join(work_dir,'exp')


bfcc_conf = path.join(conf_dir,'bfcc.conf')
dataset_conf = path.join(conf_dir,'dataset.conf')
train_conf = path.join(conf_dir,'train.conf')
dnn_conf = path.join(conf_dir,'dnn.conf')
cnn_conf = path.join(conf_dir,'cnn.conf')

## Main
if __name__ == '__main__':

    # Create directory structure
    if stage <= 0:
        if not path.exists(work_dir):
            os.mkdir(work_dir)
        if not path.exists(conf_dir):
            os.mkdir(conf_dir)
        if not path.exists(wave_dir):
            os.mkdir(wave_dir)
        if not path.exists(data_dir):
            os.mkdir(data_dir)
        if not path.exists(exp_dir):
            os.mkdir(exp_dir)

    # Dataset preparation
    if stage <= 1:
        local.prepare_scoring(f'{data_dir}/score')
        local.prepare_data(vsn_10_048, f'{data_dir}/local/tmp')
        local.format_data(f'{data_dir}/local/tmp', f'{data_dir}/local', wave_dir, channel='Audio')
        utils.segment_data(f'{data_dir}/local')
        utils.split_data_per_speaker(f'{data_dir}/local/segmented', data_dir, ['train','dev','eval'], [8,1,1])

    # Features extraction, label generation
    if stage <= 2:
        for x in ['train','dev','eval']:
            data_sub = path.join(data_dir,x)
            steps.make_bfcc(data_sub, feat_dir, bfcc_conf)
            utils.merge_scp(f'{data_sub}/bfcc.scp', file_out=f'{data_sub}/feats.scp')
            steps.make_targets(data_sub, f'{data_dir}/score', feat_dir, config=bfcc_conf)  # move to make_mfcc
            steps.make_mvn_stats(data_sub, feat_dir)

    # Model Traning
    if stage <= 3:
        data_train = path.join(data_dir,'train')
        data_dev = path.join(data_dir,'dev')
        local.train_dnn(data_train, data_dev, f'{data_dir}/score', f'{work_dir}/exp/dnn512',
            config_dataset=dataset_conf, config_mdl=dnn_conf, config_optim=train_conf)

    # Evaluation
    if stage <=4:
        data_eval = path.join(data_dir,'eval')
        steps.decode_dnn(data_eval, f'{data_dir}/score', f'{work_dir}/exp/dnn512',
            config_dataset=dataset_conf, config_mdl=dnn_conf, config_feats=bfcc_conf)
