"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
def date_to_string(date):
    return date.strftime('%Y/%m/%dT%H:%M:%S.%f')
