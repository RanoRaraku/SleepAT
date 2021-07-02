

import os
from os import path, read
import sleepat
from sleepat import io, utils


def prepare_lang(data_dir:str) -> None:
    """
    Script to prepare lang. 

    Arguments:
        
    """
    print(f'Preparing lang {data_dir}.')

    if not path.isdir(data_dir):
        print(f'Error: {data_dir} not found.')
        exit(1)

    if not path.isdir(path.join(data_dir, "lang")):
            os.makedirs(path.join(data_dir, "lang"))
    
    events = {"sil": 0,
    "eps": 1,
    "h":2,
    "e":3,
    "l":4,
    "o":5,
    "w":6,
    "r":7,
    "d":8}

    
    io.write_scp(path.join(data_dir, "lang", "events"), events)
