"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
import os
from os import path
import sleepat
from sleepat import io, utils, nnet, dataset

def train_dnn(train_data:str, dev_data:str, score_dir:str, exp_dir:str,
    config_dataset:str=None, config_mdl:str=None, config_optim:str=None):
    """
    Train a DNN Snore detection model. The dataset is a SimpleDataset
    object and the model is a DnnSnore() object. See SimpleDataset()
    for content of config_dataset. See DnnSnore for content of config_mdl.
    See train_mdl() for content of config_optim and  No kwargs transfer
    from header to any config_* to avoid confusion.
    
    Arguments:
        train_data .... directory with training data
        dev_data ... directory for development data
        score_dir ... directory that contains classes file
        exp_dir ... output directory of training
        <config_dataset> .... configuration file for SimpleDataset()
        <config_mdl> ... configuration file for DnnSnore()
        <config_optim> ... configuration file for train_mdl()
    """
    print(f'Training a DNN for snore detection to {exp_dir}.')
    if not path.isdir(exp_dir):
        os.mkdir(exp_dir)
    else:
        for file in utils.list_files(exp_dir):
            os.remove(path.join(exp_dir,file))

    classes = io.read_scp(path.join(score_dir,'classes'))
    classes_num = len(set(list(classes.values())))
    if classes_num < 1:
        print(f'Error: number of classes < 2.')
        exit(1)

    trainload = dataset.SimpleDataset(train_data,config=config_dataset).to_dataloader()
    devload = dataset.SimpleDataset(dev_data,config=config_dataset).to_dataloader()
    sample_dim = trainload.dataset.sample_dim
    if len(sample_dim) > 1:
        print(f'Error: sample dimension > 1. DNN expects a flat input.')
        exit(1)
    sample_dim = sample_dim[0]

    # Train the model
    mdl = nnet.DnnSnore(config=config_mdl, in_dim=sample_dim, out_dim=classes_num)
    nnet.train_mdl(mdl, trainload, devload, exp_dir, config=config_optim)