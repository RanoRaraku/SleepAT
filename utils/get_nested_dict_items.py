"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
def get_nested_dict_items(dictionary:dict, depth:int=0) -> tuple:
    """
    Extract keys and values from a dictionary in a particular
    depth of a nested dict.

    The function is a generator, the accumulation happens on the consumer part.
    Input:
        dictionary ... a nested dictionary (default:dict)
        depth .... depth from which to extract (default:int = 0)
    Output:
        a tuple of (key,value)
    """
    for key, value in dictionary.items():
        dpt = depth
        if isinstance(value,dict) and dpt > 0:
            dpt -=1
            yield from get_nested_dict_items(value,dpt)
        else:
            dpt = depth
            yield key,value
