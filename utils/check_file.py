"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
import os

def check_file(file:str) -> bool:
    """
    Checks if file exists.
    Input:
        file ... a path to the file
    Output:
        true/false
    """
    if os.path.isfile(file):
        return(1)
    else:
        print('File not found %s' % file)
        return(0)