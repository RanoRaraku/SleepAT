"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic IO routines.
"""
import scipy
from scipy.io import wavfile

def write_wav(file:str, fs:int, waveform):
    """
    Reads a data from a wav file. The file can be a string.
    Cannot read 24-bit wav (scipy limitation).
    Input:
        file ... string or a list of files to load
        fs ... sampling rate in Hz
        waveform ... data as numpy array
    """
    if not isinstance(file,str):
        print(f'Error write_wav(): file arg. expects string, got {type(file)}.')
        exit(1)
    if not isinstance(fs,str):
        print(f'Error write_wav(): fs arg. expects string, got {type(file)}.')
        exit(1)
    wavfile.write(file,fs,waveform)


