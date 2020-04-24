"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
import os
from os import path
import numpy as np
import torch
from torch import load
import sleepat
from sleepat import io, nnet, dataset, infer, utils

def decode_dnn(data_dir:str, score_dir:str, exp_dir:str,
    config_dataset:str=None, config_mdl:str=None, config_feats:str=None):
    """
    Decode with a DNN model saved in exp_dir. The dataset is 
    a SimpleDataset object. See SimpleDataset() for content of
    config_dataset. See DnnSnore for content of config_mdl.
    See train_mdl() for content of config_optim and  No kwargs transfer
    from header to any config_* to avoid confusion.
    
    Arguments:
        train_data .... directory with training data
        score_dir ... directory that contains classes file
        exp_dir ... output directory of training
        <config_dataset> .... configuration file for SimpleDataset()
        <config_mdl> ... configuration file for DnnSnore()
        <config_feats> ... configuration file for targets_to_annot()
    """
    print(f'Decoding {data_dir} into {exp_dir}.')
    if not path.isdir(exp_dir):
        print(f'Error: {exp_dir} does not exist.')
        exit(1)

    decode_dir = path.join(exp_dir,'decode_'+path.basename(data_dir))
    if not path.isdir(decode_dir):
        os.mkdir(decode_dir) 

    # Create a blank model and load from save
    device_fid = path.join(exp_dir,'device')
    if not path.exists(device_fid):
        print(f'Error: {device_fid} does not exist. Specify device (cuda/cpu).')
        exit(1)
    device = io.read_scp(device_fid)['device']

    classes = io.read_scp(path.join(score_dir,'classes'))
    classes_num = len(set(list(classes.values())))
    if classes_num < 1:
        print(f'Error: number of classes < 2.')
        exit(1)

    DataSet = dataset.SimpleDataset(data_dir,config=config_dataset)
    DataSet.init_decode(classes_num)
    sample_dim = DataSet.sample_dim
    if len(sample_dim) > 1:
        print(f'Error: sample dimension > 1. DNN expects a flat input.')
        exit(1)
    sample_dim = sample_dim[0]

    mdl_final = path.join(exp_dir,'final.pth')
    if not path.exists(mdl_final):
        print(f'Error: {mdl_final} does not exist.')
        exit(1)
    mdl = nnet.DnnSnore(config=config_mdl, in_dim=sample_dim, out_dim=classes_num)
    mdl.load_state_dict(torch.load(mdl_final)['model_state_dict'])
    mdl.to(device)
    mdl.eval()

    for batch in DataSet.to_dataloader():
        sample = batch['sample'].to(device)
        output = mdl(sample)
        post = torch.nn.functional.softmax(output,dim=1)
        pred = post.argmax(dim=1)
        DataSet.insert_post(post,batch['index'])
        DataSet.insert_pred(pred,batch['index'])


    # Make Transcription in the same format as Annotation
    trans = dict()
    for (utt_id,tstamp,pred) in DataSet.align_array('pred'):
        trans[utt_id] = utils.targets_to_annot(pred,classes,config_feats,
            utt_timestamp = tstamp, no_null=False)
    io.write_scp(path.join(decode_dir,'transcript'),trans)


    # Make post files which contain frame-level posteriors
    post_scp = dict()
    post_dir = path.join(decode_dir,'post')
    if not path.isdir(post_dir):
        os.mkdir(post_dir)
    else:
        for file in utils.list_files(post_dir):
            os.remove(path.join(post_dir,file))

    for (utt_id,_,post) in DataSet.align_array('post'):
        post_fid = path.join(post_dir,f'{utt_id}.post')
        io.write_npy(post_fid,post)
        post_scp[utt_id] = {'file':post_fid}
    io.write_scp(path.join(decode_dir,'post.scp'),post_scp)
