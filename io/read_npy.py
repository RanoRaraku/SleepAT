"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic IO routines.
"""
import numpy as np

def read_npy(file, concat_axis:int = 1) -> np.ndarray:
    """
    Reads a data from an npy file. The file can be a string or
    a list of strings. In case its a list, it loads them all
    and concatenates along the given axis. The standard is that
    axis=0 is time and axis=1 is a feature vector.
    Input:
        file ... string or a list of files to load
        concat_axis ... the concatenation axis for a list of array (default:int = 1)
    """
    if isinstance(file,str):
        return np.load(file)
    elif isinstance(file,list):
        if len(file) < 1:
            print('An empty list of files to load.')
            exit(1)

        data = np.load(file[0])
        if len(file) == 1:
            return data
        for fid in file[1:]:
            data = np.concatenate((data,np.load(fid)),axis=concat_axis)
        return data
    else:
        print('io.read_npy(): Wrong input type, expected string or list.')
        exit(1)
