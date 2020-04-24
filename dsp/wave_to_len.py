"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import numpy as np
import sleepat
from sleepat import io

def wave_to_len(file:str,fs:float) -> float:
    """
    Reads a data from an npy file and return waveform duration
    in seconds.
    Input:
        file ... string or a list of files to load
        fs ... sampling rate
    """
    if isinstance(file,str):
        wave = io.read_npy(file)
        return np.round(len(wave)/fs,5)
    else:
        print('utils.wave_to_len(): Wrong input type, expected string.')
        exit(1)
