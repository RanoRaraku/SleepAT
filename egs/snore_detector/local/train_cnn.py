"""
Made by Michal Borsky, 2019, copyright (C) RU
Collection of high-level routines used to built whole projects.
"""
import os
import torch
from sleepat.nnet import CnnSnore, train_mdl
from sleepat.dataset import SimpleDataset

def train_cnn(train_dir:str, dev_dir:str, eval_dir:str, exp_dir:str,
    config_dataset:str=None, config_cnn:str=None, conf_train:str=None):
    """
    Train a CNN model. The dataset is prepared accorading to SimpleDataset
    structure and the model


    Content of config_dataset. Not possible to set through kwargs here
    to avoid confusion. Done
    <left_context> ...
    <right_context> ...
    <delta_window> ...
    <delta_order> ...
    <norm_vars> ...
    config_dataset ...

    Content of config_cnn. Not possible to set throught kwargs here
    to avoid confusion.
    <>
    <>
    <>
    <>
    config_cnn ...
    """
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainload = SimpleDataset(train_dir, config=config_dataset).to_dataloader()
    devload = SimpleDataset(dev_dir, config=config_dataset).to_dataloader()
    mdl = CnnSnore(config=config_cnn).to(device=device)

    # Train the model
    train_mdl(mdl, trainload, devload, device, exp_dir, config=conf_train)

    # Eval performance
    #evalload = SimpleDataset(eval_dir,config=config_dataset).to_dataloader()
    #eval_mdl()