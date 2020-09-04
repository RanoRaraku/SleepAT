"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import torch
from torch import nn as tnet
import sleepat
from sleepat import opts

class Cnn_1c2h(tnet.Module):
    def __init__(self, config:str=None, **kwargs):
        conf = opts.Cnn_1c2h(config=config, **kwargs)
        super(Cnn_1c2h, self).__init__()
        self.network = tnet.Sequential(
            tnet.Conv2d(in_channels=1, out_channels=conf.filts, kernel_size=(3,3), stride=(1,1)),
            tnet.MaxPool2d((2,2),(2,2)),
            tnet.Dropout2d(conf.dropout_prob),
            tnet.Flatten(),
            tnet.Linear(conf.filts*11, conf.hid_dim),
            tnet.ReLU(),
            tnet.Dropout(conf.dropout_prob),
            tnet.Linear(conf.hid_dim,conf.hid_dim),
            tnet.ReLU(),
            tnet.Dropout(conf.dropout_prob),
            tnet.Linear(conf.hid_dim,conf.out_dim),
        )
    def forward(self, x):
        return self.network(x)