"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic IO routines.
"""
import os
from os import path
import json

def read_scp(file:str) -> dict():
    """
    Read an scp file, which is expected to be a json file, with a dict structure.
    See write_scp() for more info.
    Input: file ... a full path to file to read from
    Output data .... a dictionary
    """
    if not isinstance(file,str):
        print(f'Error read_scp(): file arg. expects string, got {type(file)}.')
        exit(1)
    if not path.isfile(file):
        print(f'Error read_scp(): {file} not found.')
        exit(1)

    with open(file, 'r') as fh:
        return json.load(fh)
