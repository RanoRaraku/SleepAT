"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import os
from os import path
import sleepat
from sleepat import dsp, io
from matplotlib import pyplot as plt

def align_data(data_1:str, data_2:str) -> None:
    """
    Time-align datasets using waveforms in data_1/wave and data_2/wave.
    The script assumes the waveforms are the same signal, as it uses
    cross-correlation to find appropriate time shifts "tau". The result
    is a dict of 'tau', such that data_1 is an anchor and data_2 is
    shifted by (t-tau) to match data_1. We assume the datasets use the 
    same utt_id to index files, that are ordered

    Arguments:
        vsn_dir .... folder containg VSN-10-048 dataaset.
        dst_dir .... working directory of the project
    """
    print(f'Aligning datasets {data_1} and {data_2}.')

    ## Configuration
    utt2spk_1 = io.read_scp(path.join(data_1,'utt2spk'))
    utt2spk_2 = io.read_scp(path.join(data_2,'utt2spk'))
    wave_scp_1 = io.read_scp(path.join(data_1,'wave.scp'))
    wave_scp_2 = io.read_scp(path.join(data_2,'wave.scp'))

    for utt_id_1, utt_id_2 in zip(utt2spk_1,utt2spk_2):
        if utt_id_1 != utt_id_2:
            print(f'Error: Utterance ids dont match ({utt_id_1} vs. {utt_id_2}).')
            exit(1)
        wave1 = io.read_npy(wave_scp_1[utt_id_1]['file'])
        wave2 = io.read_npy(wave_scp_2[utt_id_2]['file'])

        plt.plot(wave1)
        plt.plot(wave2)
        plt.show()

