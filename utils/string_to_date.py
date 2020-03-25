"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
from datetime import datetime

def string_to_date(string):
    return datetime.strptime(string,'%Y/%m/%dT%H:%M:%S.%f')
