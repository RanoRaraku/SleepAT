"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import numpy as np
import sleepat
from sleepat import io

def wave_to_samples(file:str) -> int:
    """
    Reads a data from an npy file and return waveform duration
    in samples.
    Input:
        file ... full path to a waveform saved as .npy
    Output
        waveform length in samples
    """
    wave = io.read_npy(file)
    return wave.shape[0]
