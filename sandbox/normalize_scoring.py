"""
Made by Michal Borsky, 2019, copyright (C) RU
Normalizes scoring for evaluation.
"""
import datetime
from datetime import timedelta
import sleepat
from sleepat import utils, objects


def normalize_scoring(scoring:list, period:dict, events:dict) -> list:
    """
    Normalize scoring for evalution by adding null events, merging events with identical
    numerical labels, and normalizing text labels. A scoring is a list of events, each is
    dict(), that occur within an utterance, which can be also empty. Full info on the 
    utterance bounds is saved in 'period'. Scoring duration is the length of that utterance
    and it must use same units as event['onset'] and event['duration']. Scoring_start is used
    to anchor events in date-time to adher to annotation file format.
    1) Add 'null' events to fill in gaps between events.
    2) Normalize text labels to eliminates ambiguous labels.
    3) Merge identical events to eliminate sequences (i.e 000->0).

    Arguments:
        scoring ... a list of events inside an utterance
        period ... info on utterance start, onset and duration
        events ... mapping between numerical and text labels of events
    Return:
        normalized scoring
    """
    scoring_dur = period['duration']
    scoring_start = period['start']

    if scoring_dur == 0:
        print(f'Error: scoring duration is 0.')
        exit(1)
    if not events:
        print(f'Error: events dict is empty.')
        exit(1)

    # 1) Adds null events to a scoring
    # 2) Solve overlapping events
    beg = 0.0    
    scoring_norm = list()
    for event in scoring:
        if event['onset'] < beg:
            event['duration'] -= (beg - event['onset'])
            event['onset'] = beg
        if event['onset'] > beg:
            onset = beg
            dur = round(event['onset']- onset,6)
            start = utils.date_to_string(utils.string_to_date(scoring_start)
                + timedelta(seconds=onset))
            scoring_norm += [{'label':'null','start':start,'onset':onset,'duration':dur}]
        scoring_norm += [event]
        beg = round(event['onset'] + event['duration'],6)

    if beg < scoring_dur:
        onset = beg
        dur = round(scoring_dur - onset,6)
        start = utils.date_to_string(utils.string_to_date(scoring_start)
            + timedelta(seconds=onset))
        scoring_norm += [{'label':'null','start':start,'onset':onset, 'duration':dur}]


    # 2) Normalize text labels    
    events_inv = dict()
    for key,val in events.items():
        if val in events_inv and key != 'null':
            continue
        events_inv[val] = key

    for event in scoring_norm:
        event['label'] = events_inv[events[event['label']]]

    # 3) Merge events with the same numerical label
    if len(scoring_norm) > 1:
        scoring_tmp = scoring_norm.copy()
        scoring_norm = list()
        event_init = scoring_tmp[0]

        for event in scoring_tmp[1:]:
            if event['label'] == event_init['label']:
                event_init['duration'] = round(event_init['duration']+event['duration'],6)
                continue
            scoring_norm.append(event_init)
            event_init = event
        scoring_norm.append(event_init)

    return scoring_norm
