"""
Made by Michal Borsky, 2019, copyright (C) RU
Train a Neural Network available in nnet module.
"""
import os
from os import path
import sleepat
from sleepat import io, utils, nnet, opts

def train_nnet(train_data:str, dev_data:str, lang_dir:str, exp_dir:str, config_mdl:str=None,
    config_ds:str=None, config_train:str=None) -> None:
    """
    Train a neural network model on train_data and using dev_data as development. It can load
    any model defined in config_mdl that exists in sleepat.nnet module, the default is Dnn4_512.
    The train_/dev_data is organized in dataset object defined in config_ds, default being
    SimpleDataset. See nnet module for alternatives. No **kwargs transfer from header to any
    config_* to avoid confusion.

    Arguments:
        train_data .... directory with training data
        dev_data ... directory for development data
        lang_dir ... directory that contains events.txt file
        exp_dir ... output directory of training

        <config_mdl> ... config file for neural network (default:str=None)
        <config_ds> .... config file for dataset (default:str=None)
        <config_train> ... config file for training (default:str=None)

    TODO:
        add more info into log from config files

    """
    # Checks and loads
    print(f'Training a Nnet on {train_data} data.')

    for item in [train_data, dev_data, lang_dir]:
        if not path.exists(item):
            print(f'Error train_nnet(): {item} not found.')
            exit(1)
    if not path.isdir(exp_dir):
        os.mkdir(exp_dir)
    else:
        for file in utils.list_files(exp_dir):
            os.remove(path.join(exp_dir,file))

    # Create Dataloader
    conf = opts.TrainNnet(config_mdl,config_ds)
    events = io.read_scp(path.join(lang_dir,'events'))
    events_num = len(set(events.values()))
    if events_num < 1:
        print(f'Error: number of events < 2 ({events_num}).')
        exit(1)

    dataset = getattr(nnet,conf.dataset)
    trainload = dataset(train_data,config=config_ds, mode='train').to_dataloader()
    devload = dataset(dev_data,config=config_ds, mode='train').to_dataloader()
    sample_dim = trainload.dataset.sample_dim

    # Train the model
    model = getattr(nnet,conf.model)
    mdl = model(config=config_mdl, in_dim=sample_dim, out_dim=events_num)
    nnet.train_torch(mdl, trainload, devload, exp_dir, config=config_train)

    print(f'Training done.')