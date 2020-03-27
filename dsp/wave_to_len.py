"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
from sleepat.io import read_npy

def wave_to_len(file:str,fs:float) -> float:
    """
    Reads a data from an npy file and return waveform duration 
    in seconds.
    Input:
        file ... string or a list of files to load
        fs ... sampling rate
    """
    if isinstance(file,str):
        wave = read_npy(file)
        return round(len(wave)/fs,5)
    else:
        print('utils.wave_to_len(): Wrong input type, expected string.')
        exit(1)
