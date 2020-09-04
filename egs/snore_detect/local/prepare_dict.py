"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import os
from os import path
import sleepat
from sleepat import io

def prepare_dict(dict_dir:str,null_events:list=['null']) -> None:
    """
    A simple lexicon used to create the targets from annotation.
    It maps between 'events' exported in .scoring.json and values
    used for training. It also contains a target value for 'null'
    label, i.e. when nothing is happening.
    """
    if not path.exists(dict_dir):
        os.makedirs(dict_dir)

    # Prepare lexicon
    events = {'snorebreath':1, 'breathing-effort':0, 'null':0}
    io.write_scp(path.join(dict_dir,'events'), events)