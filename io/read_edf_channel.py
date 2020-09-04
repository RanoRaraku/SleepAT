"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic IO routines.
"""
import pyedflib

def read_edf_channel(file:str, channel:str=None):
    """
    Reads a particular channel from an EDF file. Supports 
    returning only a single channel at a time. The sampling
    frequency is read from the header and returned as well
    Input:
        file .... a path to the edf file
        channel .... name of the signal to extract
    Output:
        (wave,fs) .... extracted wavform and sampling frequency
    """
    if not isinstance(file,str):
        print(f'{" ":3}Wrong input type, expected string.')
        exit(1)

    if not isinstance(channel,str):
        print(f'" ":3Wrong input type, expected string.')
        exit(1)

    fh = pyedflib.EdfReader(file)
    wave_idx = fh.getSignalLabels().index(channel)
    wave = fh.readSignal(wave_idx)
    fs  = fh.getSampleFrequency(wave_idx)
    return (fs,wave)