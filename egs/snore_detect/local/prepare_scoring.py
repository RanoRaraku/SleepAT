"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import os
from os import path
import sleepat
from sleepat import io

def prepare_scoring(lang_dir:str) -> None:
    """
    A simple lexicon used to create the targets from annotation.
    It maps between 'labels' and a value used for training. It also contains
    a target value for 'null' label, i.e. when nothing is happening.
    """
    if not path.exists(lang_dir):
        os.mkdir(lang_dir)

    # Prepare lexicon
    classes = {'snorebreath':1, 'breathing-effort':0, 'null':0}
    io.write_scp(path.join(lang_dir,'classes'), classes)

    # Prep classes
    #classes_int = np.unique(np.array(list(classes.values())))
    #write_scp(join(lang_dir,'classes.int'),classes_int)