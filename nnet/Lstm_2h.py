"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import torch
from torch import nn as tnet
import sleepat
from sleepat import opts

class Lstm_2h(tnet.Module):
    def __init__(self, config:str=None, **kwargs):
        """
        Optional args <> can be set from a config file or as **kwargs.
        Arguments:
            <in_dim> ... input layer dimensionality == feature_dim (def:int = 39)
            <out_dim> ... output layer dimensionality == target_dim (def:int = 2)
            <hid_dim> ... LSTM layer dimensionality (def:int = 128)
            <bottleneck_dim> ... bottleneck layer dimensionality (def:int = 40)
            <dropout_prob> ... dropout probability (def:float = 0.5)
            config ... a configuration JSON file to specify all above
            **kwargs ...setting all above through kwargs
        """ 
        self.conf = opts.Lstm_2h(config, **kwargs)
        super(Lstm_2h, self).__init__()
        self.lstm1 = tnet.LSTM(input_size=self.conf.in_dim, hidden_size=self.conf.hid_dim, num_layers=1, batch_first=True)
        self.lstm2 = tnet.LSTM(input_size=self.conf.hid_dim, hidden_size=self.conf.hid_dim, num_layers=1, batch_first=True)
        self.linear = tnet.Linear(self.conf.hid_dim,self.conf.out_dim)

    def forward(self, x):
        (x,_) = self.lstm1(x)
        (x,_) = self.lstm2(x)
        x = self.linear(x.reshape(-1,self.conf.hid_dim))
        return x