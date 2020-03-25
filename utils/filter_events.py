"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""

def filter_events(events:list, key:str='', valid_values:str='') -> list:
    """
    Filter the annotation for events that have a value for a 'key' in a list of 'valid_values'.
    Annotation is a list of events, where each event is a dictionary. The standard form for an
    event is {'label', 'onset', 'duration'}, but it can contain more 'keys'. Look at
    make_annotation() for reference. However, the routine allows to search for a general 'key'
    with a valid value.
    Input:
        events .... a list of events where each event is a dictionary
        key .... which key in the event should match the valid values (default:str = '')
        valid_values .... can be a string or a list (default:str = '')
    Output:

    """
    out = list()
    get_all = False
    if valid_values == '':
        print('Grabing everything with key %s' % key)
        get_all = True
    if isinstance(valid_values,str):
        valid_values = [valid_values]

    for event in events:
        if get_all:
            out.append(event)
            continue
        if event[key] in valid_values:
            out.append(event)
    return out
