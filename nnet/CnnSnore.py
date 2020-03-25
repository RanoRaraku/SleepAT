"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import torch.nn as tnet
from sleepat.base.opts import CnnSnoreOpts

class CnnSnore(tnet.Module):
    def __init__(self, config:str=None, **kwargs):
        conf = CnnSnoreOpts(config=config, **kwargs)
        super(CnnSnore, self).__init__()
        self.network = tnet.Sequential(
            tnet.Conv2d(in_channels=1, out_channels=conf.filts, kernel_size=(3,3), stride=(1,1)),
            tnet.MaxPool2d((2,2),(2,2)),
            tnet.Dropout2d(0.5),
            tnet.Flatten(),
            tnet.Linear(conf.filts*11, conf.neurons),
            tnet.ReLU(),
            tnet.Dropout(0.5),
            tnet.Linear(conf.neurons,conf.neurons),
            tnet.ReLU(),
            tnet.Dropout(0.5),
            tnet.Linear(conf.neurons,conf.out_dim),
        )
    def forward(self, x):
        return self.network(x)