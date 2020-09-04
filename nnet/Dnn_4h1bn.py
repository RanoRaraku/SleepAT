"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import torch
from torch import nn as tnet
import sleepat
from sleepat import opts


class Dnn_4h1bn(tnet.Module):
    """
    Simple feed forward NN with 4 hidden layers and a bottleneck layer.
    """
    def __init__(self, config:str=None, **kwargs):
        """
        Optional args <> can be set from a config file or as **kwargs.
        Arguments:
            <in_dim> ... input layer dimensionality (def:int = 39)
            <out_dim> ... output layer dimensionality (def:int = 2)
            <hid_dim> ... hidden layer dimensionality (def:int = 512)
            <dropout_prob> ... dropout probability (def:float = 0.5)
            <bottleneck_dim> ... bottleneck layer dimensionality (def:int = 40)
            config ... a configuration JSON file to specify all above
            **kwargs ...setting all above through kwargs
        """
        conf = opts.Dnn_4h1bn(config, **kwargs)
        super(Dnn_4h1bn, self).__init__()
        self.network = tnet.Sequential(
            tnet.Linear(conf.in_dim,conf.hid_dim),
            tnet.ReLU(),
            tnet.Dropout(conf.dropout_prob),
            tnet.Linear(conf.hid_dim, conf.bottleneck_dim),
            tnet.Linear(conf.bottleneck_dim,conf.hid_dim),
            tnet.ReLU(),
            tnet.Dropout(conf.dropout_prob),
            tnet.Linear(conf.hid_dim,conf.hid_dim),
            tnet.ReLU(),
            tnet.Dropout(conf.dropout_prob),
            tnet.Linear(conf.hid_dim,conf.hid_dim),
            tnet.ReLU(),
            tnet.Dropout(conf.dropout_prob),            
            tnet.Linear(conf.hid_dim,conf.out_dim),
        )
    def forward(self, x):
        return self.network(x)
