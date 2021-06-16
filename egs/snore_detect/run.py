"""
Michal Borsky, Reykjavik University, 2019.

Main script to train a snore detector. Needs to be copied,
not only linked to work properly.
"""
import os
from os import path
import sleepat
from sleepat import steps, utils, infer
from sleepat.egs.snore_detect import local


## Config section
stage = 3
vsn_10_048 = '/home/borsky/datasets/VSN-10-048/MS_scored/HDF5'
work_dir = '/home/borsky/Projects/snore_detect'
data_dir = path.join(work_dir,'data')
feat_dir = path.join(work_dir,'feats')
conf_dir = path.join(work_dir,'conf')
wave_dir = path.join(work_dir,'wave')
exp_dir = path.join(work_dir,'exp')

bfcc_conf = path.join(conf_dir,'bfcc.conf')
dataset_conf = path.join(conf_dir,'dataset.conf')
lstm_dataset_conf = path.join(conf_dir,'lstm_dataset.conf')
dnn_conf = path.join(conf_dir,'dnn.conf')
lstm_conf = path.join(conf_dir,'lstm.conf')
torch_conf = path.join(conf_dir,'torch.conf')

# Create directory structure
if stage <= 1:
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
    local.prepare_dict(f'{data_dir}/lang')
    local.prepare_vsn_10048(vsn_10_048, f'{data_dir}/local/tmp', bad_spk=['VSN_10_048_015'])
    local.format_vsn_10048(f'{data_dir}/local/tmp', f'{data_dir}/local/base', wave_dir)
    steps.segment_data(f'{data_dir}/local/base', seg_len=30)
    utils.split_data_per_speaker(f'{data_dir}/local/base/segmented', data_dir, ['train','dev','eval'], [8,1,1])

# Features extraction
if stage <= 2:
    for x in ['train','dev','eval']:
        data_sub = path.join(data_dir,x)
        steps.make_bfcc(data_sub, feat_dir, bfcc_conf)
        utils.merge_scp(f'{data_sub}/bfcc.scp', scp_out=f'{data_sub}/feats.scp')
        steps.make_targets(data_sub, f'{data_dir}/lang', feat_dir, config=bfcc_conf)
        steps.make_mvn_stats(data_sub, feat_dir)

# DNN Traning
if stage <= 3:
    data_train = path.join(data_dir,'train')
    data_dev = path.join(data_dir,'dev')
    steps.train_nnet(data_train, data_dev, f'{data_dir}/lang', f'{work_dir}/exp/dnn256',
        config_mdl=dnn_conf, config_ds=dataset_conf, config_train=torch_conf)

# DNN inference
if stage <=4:
    data_eval = path.join(data_dir,'eval')
    steps.decode_nnet(data_eval, f'{data_dir}/lang', f'{work_dir}/exp/dnn256', config_mdl=dnn_conf,
        config_ds=dataset_conf,  config_feats=bfcc_conf)


# LSTM  Traning
if stage <= 5:
    data_train = path.join(data_dir,'train')
    data_dev = path.join(data_dir,'dev')
    steps.train_nnet(data_train, data_dev, f'{data_dir}/lang', f'{work_dir}/exp/lstm_2h',
        config_mdl=lstm_conf, config_ds=lstm_dataset_conf, config_train=torch_conf)

# LSTM inference
if stage <= 6:
    data_eval = path.join(data_dir,'eval')
    steps.decode_nnet(data_eval, f'{data_dir}/lang', f'{work_dir}/exp/lstm_2h', config_mdl=lstm_conf,
        config_ds=lstm_dataset_conf,  config_feats=bfcc_conf)
