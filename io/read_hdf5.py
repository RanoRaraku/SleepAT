"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic IO routines.
"""
import h5py

def read_hdf5(file:str, dataset:str=None):
    """
    Reads a particular dataset from an hdf5 file. Supports
    returning only a single dataset at a time. The sampling
    frequency is read from the header and returned as well.
    Input:
        file .... a path to the edf file
        dataset .... name of the signal to extract
    Output:
        (wave,fs) .... extracted wavform and sampling frequency
    """
    if not isinstance(file,str):
        print(f'{" ":3}Error: read_hdf5() expects string as "file" arg., got {type(file)}.')
        exit(1)
    if not isinstance(dataset,str):
        print(f'{" ":3}Error: read_hdf5() expects string as "dataset" arg., got {type(dataset)}.')
        exit(1)

    fh = h5py.File(file,'r')
    if dataset in fh.keys():
        dset = fh[dataset]
    else:
        print(f'{" ":3}Error: {file} does not contain dataset "{dataset}".')
        print(f'{" ":3}The available datasets include:{fh.keys()}.')
        exit(1)

    return (dset.attrs['fs'],dset[()])