"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import os
from os import path
import sleepat
from sleepat import io

def prepare_dict(dict_dir:str) -> None:
    """
    A simple dictionary used to create the targets from annotation. The dictionary maps
    between events (type = str) and targets (type = int) that are used for NN training.
    """
    if not path.exists(dict_dir):
        os.makedirs(dict_dir)

    file = path.join(dict_dir,'events')
    events = {'snorebreath':1, 'breathing-effort':0, 'null':0}
    io.write_scp(file, events)