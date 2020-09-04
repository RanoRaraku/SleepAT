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
        file ... full path to a waveform saved as .npy
        fs ... sampling rate in Hz
    Output:
        waveform length in seconds        
    """
    wave = io.read_npy(file)
    return np.round(len(wave)/fs,6)
