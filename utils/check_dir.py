"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
import os

def check_dir(folder:str) -> bool:
    """
    Checks if directory exists.
    Input:
        folder ... a path to the folder
    Output:
        true/false
    """
    if os.path.exists(folder):
        return(1)
    else:
        print('Directory not foud %s' % folder)
        return(0)
