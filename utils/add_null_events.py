"""
Made by Michal Borsky, 2019, copyright (C) RU
Adds null events to a scoring.
"""
import sleepat
from sleepat import utils, objects


def add_null_events(scoring:list, period:dict) -> list:
    """
    Adds 'null' events to a scoring to fill in the gaps between events for full
    annotation. A scoring is a list of events, each is dict(), that occur within
    an utterance, which can be also empty. Full info on the utterance is saved
    within 'period'. Scoring duration is the length of that utterance and it must
    use same units as event['onset'] and event['duration']. Scoring_start is a start
    of the utterance, used to anchor events in date-time to adher to annotation file
    format.

    Arguments:
        scoring ... a list of events(dicts) for a utterance
        period ... info on utterance duration and start
    Return:
        a fully populated scoring with null events
    """
    beg = 0.0
    scoring_null = list()
    scoring_dur = period['duration']
    scoring_start = period['start']
    if scoring_dur == 0:
        print(f'Error: scoring duration is 0.')
        exit(1)

    for event in scoring:
        if event['onset'] > beg:
            onset = beg
            dur = round(event['onset']- onset,6)
            start = objects.TimeStamp(stamp=scoring_start, offset=onset)
            scoring_null += [{'label':'null','start':start.print(),'onset':onset,'duration':dur}]
        scoring_null += [event]
        beg = round(event['onset'] + event['duration'],6)

    if beg < scoring_dur:
        onset = beg
        dur = round(scoring_dur - onset,6)
        start = objects.TimeStamp(stamp=scoring_start, offset=onset)
        scoring_null += [{'label':'null','start':start.print(),'onset':onset, 'duration':dur}]

    return scoring_null
