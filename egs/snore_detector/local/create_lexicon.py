"""
Made by Michal Borsky, 2019, copyright (C) RU.
"""
from sleepat.io import write_scp

def create_lexicon(lexicon_file:str) -> None:
    """
    A simple lexicon used to create the targets from annotation.
    It maps between 'labels' and a value used for training. It also contains
    a target value for 'null' label, i.e. when nothing is happening.
    Input:
        lexicon_file .... a path to lexicon file to create
    """
    lex = dict()
    lex['snorebreath'] = 1
    lex['breathing-effort'] = 2
    lex['null'] = 0
    write_scp(lexicon_file, lex)