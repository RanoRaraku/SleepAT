"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
def filter_scoring(scoring:list, key:str='', valid_values:str=['']) -> list:
    """
    Filter the scoring for events that have a value for a 'key' in a list of 'valid_values'.
    Scoring is a list of events, where each event is a dictionary. The standard form for
    an event is {'label', 'onset', 'duration'}, but it can contain more 'keys'. Look at
    make_annotation() for reference. However, the routine allows to search for a general
    'key' with a valid value.

    Arguments:
        scoring .... a list of events where each event is a dictionary
        key .... key to match with valid values (default:str = '')
        valid_values .... can be a string or a list (default:str = [''])
    Output:

    """
    out = list()
    if valid_values == ['']:
        print('Grabing everything with key %s' % key)
        return scoring
    if isinstance(valid_values,str):
        valid_values = [valid_values]

    for event in scoring:
        if event[key] in valid_values:
            out.append(event)
    return out
