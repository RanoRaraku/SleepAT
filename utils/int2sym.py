"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
def int2sym(symbols:dict, value, prefer_null:bool=True) -> str:
    """
    Transforms values according to symbol dictionary. In case
    of collision between symbols, return 1st symbol found. By
    default, null symbols override other conflicting symbols.

    Arguments:
        symbols .. a nested dictionary (default:dict)
        value .... integer to transform
    Output:
        a tuple of (key,value)
    """
    if not symbols:
        print(f'Error: symbol table is empty.')
        exit(1)

    symbols_inv = dict()
    for key, item in symbols.items():
        if item in symbols_inv and key != 'null':
            continue
        symbols_inv[item] = key
    return symbols_inv[value]