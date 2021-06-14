"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic IO routines.
"""
import h5py

def read_hdf5(file:str, dataset:str=None):
    """
    Reads a particular dataset from an hdf5 file. Supports returning only a single
    dataset at a time. The sampling rate is read from the header and returned.

    Arguments:
        file .... a path to the edf file
        dataset .... name of the signal to extract
    Output:
        (fs,wave) .... extracted wavform and sampling frequency if dataset specified
        {dataset: wave} .... a full dictionary with all datasets if dataset == None
    """
    if not isinstance(file, str):
        print(f'Error read_hdf5(): "file" expects string, got {type(file)}.')
        exit(1)   
    fh = h5py.File(file,'r')

    
    if dataset == None:
        out = fh
    elif isinstance(dataset, str):
        if dataset in fh.keys():
            dset = fh[dataset]
            out = (dset.attrs['fs'],dset[()])
        else:
            print(f'Error: {file} does not contain dataset {dataset}.')
            print(f'The available datasets are:{fh.keys()}.')
            exit(1)
    else:
        print(f'Error read_hdf5(): expects string as "dataset" arg., got {type(dataset)}.')
        exit(1)


    return out