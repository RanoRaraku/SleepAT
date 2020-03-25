"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
import numpy as np
from sleepat.io import read_npy

def wave_to_len(file:str,fs:float):
    """
    Reads a data from an npy file. The file can be a string or
    a list of strings, see read_npy(). The return value is wave-
    form length in seconds.
    Input:
        file ... string or a list of files to load
        fs ... sampling rate (default:float = 8000)
    """
#    conf = WaveToLenOpts(config, **kwargs)

    if isinstance(file,str):
        wave = read_npy(file)
        return np.round(len(wave)/fs,5)
    else:
        print('utils.get_wave_len(): Wrong input type, expected string.')
        exit(1)
