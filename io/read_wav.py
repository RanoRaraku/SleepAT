"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic IO routines.
"""
import os
import scipy
from scipy.io import wavfile

def read_wav(file:str):
    """
    Reads a data from a wav file. The file can be a string.
    Cannot read 24-bit wav (scipy limitation).
    Input:
        file ... string or a list of files to load
    Output:
        (fs,wave) ... sampling rate in Hz and waveform
    """
    if not isinstance(file,str):
        print(f'Error read_wav(): expects string, got {type(file)}.')
        exit(1)
    if not os.path.isfile(file):
        print(f'Error read_wav(): file {file} not found.')
        exit(1)
    return(wavfile.read(file))




