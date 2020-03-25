"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
import os

def setup_workdir(work_dir:str):
    """
    Setup basic structure of a working directory.
    """
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    data = os.path.join(work_dir,'data')
    if not os.path.exists(data):
        os.mkdir(data)
    conf = os.path.join(work_dir,'conf')
    if not os.path.exists(conf):
        os.mkdir(conf)
    exp = os.path.join(work_dir,'exp')
    if not os.path.exists(exp):
        os.mkdir(exp)
    feats = os.path.join(work_dir,'feats')
    if not os.path.exists(feats):
        os.mkdir(feats)
    wave = os.path.join(work_dir,'wave')
    if not os.path.exists(wave):
        os.mkdir(wave)
