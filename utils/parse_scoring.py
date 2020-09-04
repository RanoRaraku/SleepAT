"""
Made by Michal Borsky, 2019, copyright (C) RU
"""
import sleepat
from sleepat import io

def parse_scoring(file:str, scoring = ['ms_snore','ms_snore_v2']):
    """
    Loads a *.scoring.json that contains raw NDF exports. The 'scoring' is
    top level key and a list of dictionaries is the value. The file can
    contain multiple scorings, each with its unique name. The processing
    outputs a list of dictionaries that contains: {'label', 'onset', 'duration'}.
    This is a standardized format that is expected to create the annotation
    file. This can be directly saved, or it can be further filtered using
    "filter_annot()" for specific events, have the onsets aligned etc.
    The events are sorted chronologically by their onset, but not their duration.

    Input:
        file .... a path to JSON file that contains scorings.
        scoring .... name of scorings to export (default:list = ['ms_snore','ms_snore_v2]').
    Output:
        event_list ... a standardized dictionary of all scorings
    """
    out = list()
    data = io.read_scp(file)
    start_marker = 'period-analysisstart'
    end_marker = 'period-analysisstop'

    ## Some checks
    if not scoring in data:
        print('Warning: Scoring %s does not exist for file %s' % (scoring,file))
        return out
    events = data[scoring]
    if not events:
        print('Warning: Scoring %s contains no events to export.' % scoring)
        return out

    ## Main - merge period-analysisstart and period-analysisstop into 1 event
    for i,event in enumerate(events):
        tmp = {'label': event['Event_Type'],'start':event['Start'], 'onset': event['Onset'], 'duration': event['Duration']}
        if tmp['label'] == start_marker:
            tmp['label'] = 'analysis-period'
            idx = i
        if tmp['label'] == end_marker:
            out[idx]['duration'] = round(tmp['onset'] - out[idx]['onset'],5)
        else:
            out.append(tmp)
    return out
