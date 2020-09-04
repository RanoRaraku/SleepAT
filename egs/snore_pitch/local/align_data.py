"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import os
from os import path
import scipy
from scipy import signal
import numpy as np
import sleepat
from sleepat import dsp, io
from matplotlib import pyplot as plt

def align_data(data_1:str, data_2:str, config:str=None, **kwargs) -> None:
    """
    Time-align datasets using waveforms in data_1/wave and data_2/wave.
    The script assumes the waveforms are the same signal, as it uses
    cross-correlation to find appropriate time shifts "tau". The result
    is a dict of 'tau', such that data_1 is an anchor and data_2 is
    shifted by (t-tau) to match data_1. We assume the datasets use the 
    same utt_id to index files, that are ordered.

    Arguments:
        vsn_dir .... folder containg VSN-10-048 dataaset.
        dst_dir .... working directory of the project
    """
    print(f'Aligning datasets {data_1} and {data_2}.')
    #conf = opts.AlignData(config=config, **kwargs)

    ## Configuration
    utt2spk_1 = io.read_scp(path.join(data_1,'utt2spk'))
    utt2spk_2 = io.read_scp(path.join(data_2,'utt2spk'))
    scp_1 = io.read_scp(path.join(data_1,'wave.scp'))
    scp_2 = io.read_scp(path.join(data_2,'wave.scp'))

    for utt_1, utt_2 in zip(utt2spk_1,utt2spk_2):
        if utt_1 != utt_2:
            print(f'Error: Utterance ids dont match ({utt_1} vs. {utt_2}).')
            exit(1)

        (file_1, fs_1) = scp_1[utt_1].values()         
        (file_2, fs_2) = scp_2[utt_2].values()
        (sig_1, sig_2) = io.read_npy(file_1), io.read_npy(file_2)

        # Signals have different 'fs' since they used different sensors
        if fs_1 > fs_2:
            k = int(fs_1/fs_2*len(sig_2))
            sig_2 = signal.resample(sig_2,k)
        if fs_2 > fs_1:
            k = int(fs_2/fs_1*len(sig_1))
            sig_1 = signal.resample(sig_1,k)

        # The recording is not perfect, there are often missing datapoints
        (slen_1, slen_2) = len(sig_1),len(sig_2)        
        if slen_1 > slen_2:
            sig_1 = sig_1[:slen_2]
        if slen_2 > slen_1:
            sig_2 = sig_2[:slen_1]

        tau_range = 1000
        wlen = 10000
        loss = np.zeros(shape=(2*tau_range+1,1),dtype=np.float32)
        x = sig_1[wlen:-wlen]
        for tau in range(-tau_range,tau_range+1):
            y = sig_2[wlen+tau:-wlen+tau]
            loss[tau+tau_range] = ((x - y)**2).sum()
        shift = np.argmin(loss)-tau_range
        print(f'{utt_1} and shift {shift}.')
        