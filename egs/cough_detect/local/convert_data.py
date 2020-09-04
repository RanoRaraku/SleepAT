"""
Michal Borsky, Reykjavik University, 2019.
"""
import os
from os import path
import cbor2
import h5py
import numpy as np
import sleepat
from sleepat import io, utils

## Config section
folder = '/home/borsky/datasets/Cough_Internet/backup'
folder_2 = '/home/borsky/datasets/Cough_Ossur/backup'


## Main
"""
i = 0
for file in utils.list_dir(folder):
    dir_out = path.dirname(path)

    if 'noise' in file:
        continue
    if 'json' in file:
        i += 1
        a = io.read_scp(f'{folder}/{file}')
        fs = int(1000*(1/a['payload']['interval_ms']))
        wave = a['payload']['values']
        wave = np.array(wave,np.float32)
        if min(wave) < -32768 or max(wave) > 32767:
            print(f'{file} out of bounds.')
        name = 'SA{0:03}_000'.format(i)
        io.write_wav(f'{folder2}/{name}.wav',fs,wave)  
    if 'wav' in file:
        (fs,wave)= io.read_wav(f'{folder}/{file}')
        i += 1
        for j in range(2):
            name = 'SA{0:03}_00{1}'.format(i,j)
            io.write_wav(f'{dir_out}/{name}.wav',fs,wave[:,j])

i = 0
for file in utils.list_dir(folder):
    dir_out = path.dirname(path)

    if 'noise' not in file:
        continue
    if 'cbor' in file:
        with open(f'{folder}/{file}', 'rb') as fp:
            obj = cbor2.load(fp)
    if 'json' in file:
        obj = io.read_scp(f'{folder}/{file}')
    i += 1
    fs = int(1000*(1/obj['payload']['interval_ms']))
    wave = obj['payload']['values']
    wave = np.array(wave,np.float32)
    if min(wave) < -32768 or max(wave) > 32767:
        print(f'{file} out of bounds.')
    name = 'NA{0:03}_000'.format(i)
    io.write_wav(f'{dir_out}/{name}.wav',fs,wave)
"""

dir_out = path.dirname(folder_2)
fid = h5py.File(f'{dir_out}/Cough_Ossur.hdf5','w')
for file in utils.list_files(folder_2,'wav'):
    (fs,wave)= io.read_wav(f'{folder_2}/{file}')
    name = f'Ossur_000'
    dset = fid.create_dataset(name,data=wave[:,0])
    dset.attrs['fs'] = fs
