"""
Made by Michal Borsky, 2019, copyright (C) RU.
Basic utils routine.
"""
import os

def list_files(directory:str, extension:str='') -> list():
    """
    Searches directory for files with a specified extension. Only the
    file names are returned, not fully specified path.
    Input:
        directory .... a directory to search
        extentsion .... extension to filter, leave blank for everything.
    Output:
        file_list ... a list of files
    Use:

    """
    if not isinstance(directory,str):
        print('Wrong input type, expected string.')
        exit(1)

    if not isinstance(extension,str):
        print('Wrong input type, expected string.')
        exit(1)

    file_list = list()
    for file in os.listdir(directory):
        if not os.path.isfile(os.path.join(directory,file)):
            continue
        if file.endswith(extension):
            file_list.append(file)
    file_list.sort()
    return file_list
