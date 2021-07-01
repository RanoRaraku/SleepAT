"""
Michal Borsky, Reykjavik University, 2020.

Main script to evaluate scoring reliability between a pair scorers
using methodology as defined for a detection problem in machine learning.
"""
import os
from os import path
import sleepat
from sleepat import steps, utils, infer, io
from sleepat.egs.ctc import local

## Config section
stage = 0
audio_data = '/home/borsky/Projects/CTC-Friday/audio'
work_dir = '/home/borsky/Projects/CTC-Friday'
data_dir = path.join(work_dir,'data')
feat_dir = path.join(work_dir,'feats')
exp_dir = path.join(work_dir,'exp')
wave_dir = path.join(work_dir,'wave')


if stage <= 0:
    if not path.exists(wave_dir):
        os.mkdir(wave_dir)
    if not path.exists(data_dir):
        os.mkdir(data_dir)
    if not path.exists(exp_dir):
        os.mkdir(exp_dir)


if stage <= 1:
    local.prepare_data(audio_data, data_dir, wave_dir)
    #local.prepare_lang(data_dir)

if stage <= 2:
    for subset in ['training','test']:
        steps.make_mfcc(path.join(data_dir,subset),feat_dir)
        steps.make_mvn_stats(path.join(data_dir,subset),feat_dir)

