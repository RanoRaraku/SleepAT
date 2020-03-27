"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
from os import mkdir
from os.path import join, exists
import sleepat.io as io

def prepare_scoring(lang_dir:str) -> None:
    """
    A simple lexicon used to create the targets from annotation.
    It maps between 'labels' and a value used for training. It also contains
    a target value for 'null' label, i.e. when nothing is happening.
    """
    if not exists(lang_dir):
        mkdir(lang_dir)

    # Prepare lexicon
    classes = {'snorebreath':1, 'breathing-effort':0, 'null':0}
    io.write_scp(join(lang_dir,'classes'), classes)

    # Prep classes
    #classes_int = np.unique(np.array(list(classes.values())))
    #write_scp(join(lang_dir,'classes.int'),classes_int)