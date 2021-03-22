"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import os
from os import path
import sleepat
from sleepat import io, utils

def prepare_data(src_dir:str, dst_dir:str, wave_dir:str) -> None:
    """
    Script to prepare Cough_Internet dataset to create wav.scp, annotation,
    utt2spk, spk2utt files. Everything is indexed by 'utt_id'. Scorings are
    done manually. Everything is dumped into a dst_dir.
    Input:
        vsn_dir .... folder containg VSN-10-048 dataaset.
        dst_dir .... working directory of the project
    """
    print(f'Preparing dataset {src_dir}.')

    ## Configuration
    if not path.isdir(src_dir):
        exit(1)
    if not path.isdir(dst_dir):
        os.makedirs(dst_dir)
    if not path.isdir(wave_dir):
        os.makedirs(dst_dir)


    ## Create the wave.scp file (we extract wave later)
    ## all speakers are bundled inside hdf5 with dataset
    ## names as speaker_id
    waves, utt2spk, annot = dict(), dict(), dict()
    for fid in utils.list_files(src_dir,'.hdf5'):
        hdf5_file = path.join(src_dir,fid)
        scoring = io.read_scp(path.join(src_dir, fid[:-5]+'.scoring.json'))

        for utt_id in utils.list_hdf5(hdf5_file):
            spk_id = utt_id[:5]
            fs,waveform = io.read_hdf5(hdf5_file,utt_id)
            npy_file = path.join(wave_dir, utt_id + '.npy')
            waves[utt_id] = {'file':npy_file,'fs':int(fs)}
            utt2spk[utt_id] = spk_id
            io.write_npy(npy_file, waveform)

            # Annotation file from scoring.json
            annot[utt_id] = scoring[utt_id]
    spk2utt = utils.utt2spk_to_spk2utt(utt2spk)

    # Dump on disk
    io.write_scp(path.join(dst_dir,'wave.scp'),waves)
    io.write_scp(path.join(dst_dir,'utt2spk'),utt2spk)
    io.write_scp(path.join(dst_dir,'spk2utt'),spk2utt)
    io.write_scp(path.join(dst_dir,'annotation'),annot)
