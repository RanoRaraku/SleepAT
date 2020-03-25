"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic IO routines.
"""
import os

def list_files(directory:str, extension:str='') -> list():
    """
    Searches directory for files with a specified extension.
    Input:
        dir .... a directory to search
        extentsion .... extension to search for. Leave blank for everything.
    Output:
        file_list ....
    """
    if not isinstance(directory,str):
        print('Wrong input type, expected string.')
        exit(1)

    if not isinstance(extension,str):
        print('Wrong input type, expected string.')
        exit(1)

    file_list = list()
    for file in os.listdir(directory):
        if file.endswith(extension):
            file_list.append(file)
    file_list.sort()
    return file_list