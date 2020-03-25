"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import numpy as np
from os.path import join
from sleepat.io import write_scp

def prepare_scoring(lang_dir:str) -> None:
    """
    A simple lexicon used to create the targets from annotation.
    It maps between 'labels' and a value used for training. It also contains
    a target value for 'null' label, i.e. when nothing is happening.
    """

    # Prepare lexicon
    classes = dict()
    classes['snorebreath'] = 1
    classes['breathing-effort'] = 0
    classes['null'] = 0
    write_scp(join(lang_dir,'classes'), classes)

    # Prep classes
    classes_int = np.unique(np.array(list(classes.values())))
    write_scp(join(lang_dir,'classes.int'),classes_int)