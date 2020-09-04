"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic IO routines.
"""
import os
from os import path
import numpy as np

def read_npy(file:str) -> np.ndarray:
    """
    Reads a data from an npy file.
    Input:
        file ... string or a list of files to load
    Output:
        numpy array
    """
    if not isinstance(file,str):
        print(f'Error read_npy(): file arg. expects string, got {type(file)}.')
        exit(1)
    if not path.isfile(file):
        print(f'Error read_npy(): file {file} not found.')
        exit(1)
    return np.load(file)
