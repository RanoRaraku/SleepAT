"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic IO routines.
"""
import numpy as np

def write_npy(file:str, data:np.ndarray=None, dtype:str='float32') -> None:
    """
    Write a numpy array into a npy_file. Used to store features and targets.
    Features for each utterance are being stored in a separte file. Used to
    make sure data storage is consistent. It is possible to save data in
    whatever data type numpy supports. See documentation for more information. 
    Input:
        file .... a npy file to store the data into.
        data .... a numpy array to be stored.
        dtype .... a data type for saving (default: str = 'float32')
    """
    if not isinstance(file,str):
        print('Wrong input type, expected string.')
        exit(1)

    if not isinstance(data,np.ndarray):
        print('io.write_npy(): Wrong input type, expected np.ndarray.')
        exit(1)

    np.save(file, data.astype(dtype))