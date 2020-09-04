"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import torch
from torch import nn as tnet
import sleepat
from sleepat import opts


class Dnn_2h(tnet.Module):
    """
    Simple feed forward NN optimized for snore detection.
    """
    def __init__(self, config:str=None, **kwargs):
        """
        Optional args <> can be set from a config file or as **kwargs.
        Arguments:
            <in_dim> ... input layer dimensionality (def:int = 39)
            <out_dim> ... output layer dimensionality (def:int = 2)
            <hid_dim> ... hidden layer dimensionality (def:int = 512)
            <hid_layers> ... no. neurons in hidden layer (def:int = 2) (not used)
            <dropout_prob> ... dropout probability (def:float = 0.5)
            config ... a config. file to set optional args. <> (default:str=None)
            **kwargs ... optional args. <>
        """
        conf = opts.Dnn_2h(config, **kwargs)
        super(Dnn_2h, self).__init__()
        self.network = tnet.Sequential(
            tnet.Linear(conf.in_dim,conf.hid_dim),
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
