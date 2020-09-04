"""
Made by Michal Borsky, 2019, copyright (C) RU
Train a Gaussian model for duration.
"""
import os
from os import path
import numpy as np
import sleepat
from sleepat import io, utils

def train_dur(train_data:str, lang_dir:str, exp_dir:str, config_mdl:str=None) -> None:
    """
    Train a duration model on train_data. It can load any model defined in config_mdl that
    exists in sleepat.mdl module. Proof of concept.

    Arguments:
        train_data .... directory with training data
        lang_dir ... directory that contains events.txt file
        exp_dir ... output directory of training
        <config_mdl> ... config file for neural network (default:str=None)
    """
    print(f'Training duration model {train_data} data.')

    if not path.isdir(exp_dir):
        os.mkdir(exp_dir)
    annot = io.read_scp(path.join(train_data,'annot'))
    periods = io.read_scp(path.join(train_data,'periods'))
    events = io.read_scp(path.join(lang_dir,'events'))
    events_num = len(set(events.values()))

    a,b = list(),list()
    for utt_id,scoring in annot.items():
        scoring = utils.normalize_scoring(scoring,periods[utt_id],events)
        snores = utils.filter_scoring(scoring,'label','snorebreath')
        null = utils.filter_scoring(scoring,'label','null')
        a += [i['duration'] for i in snores]
        b += [i['duration'] for i in null]

    a = np.array(a)
    b = np.array(b)
    gmm = {'snore': {'mu': a.mean(),'std':a.std()}, 'null': {'mu': b.mean(),'std':b.std()}}
    io.write_scp(path.join(exp_dir,'dur.mdl'),gmm)