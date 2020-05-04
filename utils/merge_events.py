"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import sleepat

def merge_events(events:list, event_a:str, event_b:str) -> list:
    """
    Merge event_a and event_b for all their occurances in a list
    of events. 

    Arguments:
        events ... a path to json annotation file that contains scorings.
        event_a ...
        event_b
    Output:
        events ... list of events where event_a and event_b are merged
    """
    out = list()


    ## Main - merge event_a and event_b into 1 event
    for i,event in enumerate(events):
        tmp = {'label': event['Event_Type'],'start':event['Start'], 'onset': event['Onset'], 'duration': event['Duration']}
        if tmp['label'] == event_a:
            tmp['label'] = 'analysis-period'
            idx = i
        if tmp['label'] == event_b:
            out[idx]['duration'] = round(tmp['onset'] - out[idx]['onset'],5)
        else:        
            out.append(tmp)
    return out