"""
Made by Michal Borsky, 2019, copyright (C) RU
Normalizes scoring for evaluation.
"""
import datetime
from datetime import timedelta
import sleepat
from sleepat import utils

def synchronize_scorings(*args) -> list:
    """
    Synchronize scoring_1 and scoring_2 to have equal onset and duration. Timestamps and onsets
    of all events are adjusted. A scoring is a list of events that occur within an utterance,
    but can be also empty. Full info on the utterance bounds is saved in period_1/2. Scoring
    duration is the length of that utterance and it must use same units as event['onset'] and
    event['duration']. The scorings and periods are changed, not copied.

    Arguments:
        *args ... a list of tuple, where each tuple is a (scoring,period)
    Return:
        (scoring_1, scoring_2, period_1, period_2)

    Note:
    BS doesnt work post feat extraction atd, must be done before any training/feat extract
    """
    #Checks
    if len(args) < 1:
        print(f'Warning: a single (scoring,period) supplied.')
        return(args[0])

    # Find bounds of new period
    end = 0.0
    onset = float('inf')
    for idx, pair in enumerate(args):
        if len(pair) != 2:
            print(f'Error: tuple at index {idx} containes > 2 elements {len(pair)}.')
        period = pair[1]
        if not isinstance(period,dict):
            print(f'Error: period at index {idx} is not a dictionary {type(period)}.')
            exit(1)

        if period['onset'] < onset:
            onset = period['onset']
        if period['onset'] + period['duration'] > end:
            end = period['onset'] + period['duration']
            
    # Adjust event bounds to new period
    for idx, pair in enumerate(args):
        (scoring, period) = pair
        if not isinstance(scoring, list):
            print(f'Error: scoring at index {idx} is not a list {type(scoring)}.')
            exit(1)        

        if period['onset'] > onset:
            shift = period['onset'] - onset
            period['onset'] = onset
            period['start'] = utils.date_to_string(utils.string_to_date(period['start'])
                    - timedelta(seconds=shift))           
            for event in scoring:
                event['onset'] = round(event['onset'] + shift,6)

        if period['onset'] + period['duration'] < end:
            period['duration'] = round(end - onset,6)

    # This return is BS as well
    return args[0][0],args[0][1],args[1][0],args[1][1]