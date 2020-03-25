"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
from sleepat.io import write_scp

def create_conf(conf_file:str, **kwargs) -> None:
    """
    A general function to create a configuration file, lexicon, etc. A config file is a dictionary
    saved as JSON. It is here to show the structure. Often its easier to write it by hand.
    Input:
        conf_file .... a path to conf file to create
        **kwargs .... of dict of all variables and values
    """
    conf = dict()
    for key, value in kwargs.items():
        conf[key] = value
    write_scp(conf_file, conf)
