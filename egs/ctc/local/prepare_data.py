"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import os
from os import path, read
import sleepat
from sleepat import io, utils

def prepare_data(data_dir:str, dst_dir:str, wave_dir:str, config:str=None, **kwargs) -> None:
    """
    Script to prepare data.

    Arguments:
        data_dir ... folder containg VSN-10-048 dataset.
        dst_dir ... working directory of the project
        wave_dir ... folder to save npy extracted from WAV
        <utt2seg> ... segmentation dict if used_period = 'manual'
        config ... config file to set optional args <>, (default:str=None)
        **kwargs ... to set optional args (<>)
    """
    print(f'Preparing dataset {data_dir} into {dst_dir}.')

    if not path.isdir(data_dir):
        print(f'Error: {data_dir} not found.')
        exit(1)
    if not path.isdir(dst_dir):
        os.makedirs(dst_dir)


    for subset in ['training','test']:

        if not path.isdir(path.join(dst_dir,subset)):
            os.makedirs(path.join(dst_dir,subset))
        if not path.isdir(path.join(wave_dir,subset)):
            os.makedirs(path.join(wave_dir,subset))


        (wave,utt2spk,annot) = dict(), dict(), dict()
        for file in utils.list_files(path.join(data_dir,subset),'wav'):

            utt_id = file.split('.')[0]
            spk_id = file.split('_')[0]

            # Recast original WAV into numpy arrays
            npy_file = path.join(wave_dir,subset,utt_id+".npy")
            (fs,waveform) = io.read_wav(path.join(data_dir,subset,file))
            io.write_npy(npy_file, waveform)

            # Pupulate standard
            wave[utt_id] = {'file':npy_file,'fs':fs}
            utt2spk[utt_id] = spk_id
            annot[utt_id] = "hello world"

        spk2utt = utils.utt2spk_to_spk2utt(utt2spk)
        io.write_scp(path.join(dst_dir,subset,'wave.scp'),wave)
        io.write_scp(path.join(dst_dir,subset,'utt2spk'),utt2spk)
        io.write_scp(path.join(dst_dir,subset,'spk2utt'),spk2utt)
        io.write_scp(path.join(dst_dir,subset,'annot'),annot)
