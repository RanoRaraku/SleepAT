"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic utils routine.
"""
import os
import h5py

def list_hdf5(file:str) -> list():
    """
    Searches directory for files with a specified extension.
    Input:
        dir .... a directory to search
        extentsion .... extension to search for. Leave blank for everything.
    Output:
        file_list ....
    """
    if not isinstance(file,str):
        print(f'{" ":3}Error list_hdf5(): expects string as "file" arg., got {type(file)}.')
        exit(1)

    if not os.path.isfile(file):
        print(f'{" ":3}Error list_hdf5(): {file} not found.')
        exit(1)

    fh = h5py.File(file,'r')
    return fh.keys()