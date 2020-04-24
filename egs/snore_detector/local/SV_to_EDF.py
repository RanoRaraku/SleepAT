"""
Made by Michal Borsky, Reykjavik University, 2019


Script to convert SV database to EDF+. Collection of local routines specific to VSN-10-048 dataset
to process it into a standard format. All further script process the dataset on an utterance-basis.
The 'utt_id' is THE identifier, it needs to be unique and the same across all files. The identifier
is always the top-level key for all lists, which are saved as *.scp saved as json files. The 
processed dataset will contain the following files:
    annotation ... contains all events of interest.
    wave.scp ...contains paths to waveform files.
    utt2spk ... utterance_id to speaker_id mapping
    spk2utt ... speaker_ids to utterance_ids mapping
    segments ...contains info how to segment other files, optional.
"""

# Script to convert SV database to EDF+
# Made by Michal Borsky, Reykjavik University, 2019, 
# Distributed under **** licence
#
# Notes: 
# 1) Contrary to EDF specification, the NDF signal values are exported as is (data type & range).
# 2) The digital_min and digital_max in EDF header contains placeholder values -32768 and 32767.
# 3) The EDF.Startdatetime format does not allow miliseconds (:%f) so we round up NDF.period.start to nearest integer.
# 4) NDF signal contains unit but its not exported by the API, would need to rework API to export it.
# 5) The script works OK on VSN-10-048 and VSN-11-121 datasets. It's probably general enough to work on any NDF recording.
# 6) The EDF specification requires data is saved in a specified order:  write_other() -> write_signals() -> write_annotation()
# 7) It is possible to supply your own list of signals to be exported. Otherwise it will export all valid signals.
#-------------------------------------------------------------------------------------------------------------------
# ToDo:
#-------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import pyedflib
import json
import nox_reader as nr
from datetime import timedelta, datetime
from matplotlib import pyplot as plt

def date_to_string(timestamp):
    return timestamp.strftime('%Y/%m/%dT%H:%M:%S.%f')

def get_bad_signals():
    """
    A list of bad/useless signals that will not be extracted. Compiled from VSN-10-048 and VSN-11-121 datasets.
    Used later in get_valid_signals().
    """
    list = ['Audio Volume','Audio Volume (A)', 'Audio Volume (db)','Right Audio Volume','Left Audio Volume', 'Breath Quality (5s)',
    'Breath Quality (10s)','Breath Quality (20s)','Thorax Contribution']
    return list

def get_valid_signals(recording,period):
    """
    Check if recording lists only valid and existing signals. A valid signal means {for all x in signal: x != 0}.
    Splits the period into 1 hour chunks since {if there exists segment: is_valid(segment) then is_valid(signal)}.
    Needs to be done before an EDF is opened as we need list of valid signals. Bad signals automatically excluded.
    Input:
        recording .... an object of "nox_reader.nox_recording_class.Recording" type. Contains everything about the NDF.
        period .... an object of "nox_reader.nox_recording_class.Period" type. Defines the begining and end of exported waveforms.
    Output:
        signals_valid .... list of valid signals
    """
    bad_signals = get_bad_signals()
    signals = recording.get_available_signal_labels_for_recording()
    signals_valid = []
    if period.duration > 3600:
        segments = segment_period(period)
    else:
        segments = [period]    

    for signal in signals:
        if signal in bad_signals:
            continue
        for segment in segments:
            chunk = recording.get_signal(signal, segment)
            if np.any(chunk.data):
                signals_valid.append(signal)
                break
    return signals_valid

def write_period_file(periods,file):
    """
    Extract periods contained in "file". Each line contains one entry, format is "panelID start stop".
    Start_string and stop_string have the format of "year/month/day{T}hour:minute:second.millisecond". If file does not exist,
    return empty dictionary. The output is a dict where key is panel and value is a "nnox_reader.nox_recording_class.Period" object.
    Input : file .... a file containing a list of panels with corresponding periods
    Output : period_dict .... a dictionary of panelIDs and periods found in file
    """
    fh = open(file,'w')
    for panel, period in periods.items():
        start = date_to_string(period.start)
        stop = date_to_string(period.stop)
        if start > stop:
            print("Error: period starts after it ends.")
            continue
        fh.write(f'{panel} {start}-{stop}\n')
    fh.close()
    
def read_period_file(file):
    """
    Extract periods contained in "file". Each line contains one entry, format is "panelID start stop".
    Start_string and stop_string have the format of "year/month/day{T}hour:minute:second.millisecond". If file does not exist,
    return empty dictionary. The output is a dict where key is panel and value is a "nnox_reader.nox_recording_class.Period" object.
    Input : file .... a file containing a list of panels with corresponding periods
    Output : period_dict .... a dictionary of panelIDs and periods found in file
    """
    period_dict = {}
    if not os.path.isfile(file):
        print('Warning could not find file: {}'.format(file))
        return period_dict

    fh = open(file,'r')
    for line in fh:
        panel, period = line.rstrip().split(' ')
        startStr, stopStr = period.rstrip().split('-')
        start = datetime.strptime(startStr,'%Y/%m/%dT%H:%M:%S.%f')
        stop = datetime.strptime(stopStr,'%Y/%m/%dT%H:%M:%S.%f')
        if start > stop:
            print("Wrong entry at line:",line," Period starts after it ends.")
            continue
        period_dict[panel] = nr.nrcPeriod(start,stop)
    return period_dict

def get_valid_period(recording, panel, period_dict):
    """
    Obtain a valid recording period for a particular panel. Entry inside period_dict takes priority over information inside
    recording object. Used mainly because "recording.whole_recording_period" is often wrong(too long) and it results in 
    extracting a long series of zeros. It can be GB of null data.
    Input:
        recording .... an object of "nox_reader.nox_recording_class.Recording" type. Contains everything about the NDF.
        panel .... a recoring ID. Must match a key in period_dict, otherwise not used
        period_dict .... a dictionary of panelIDs and periods, created by calling read_period_file()
    Output:
        period .... a valid period for the recording, an "nox_reader.nox_recording_class.Period" object
    """
    if panel in period_dict:
        period = ceil_periodStart(period_dict[panel])
        print('Found valid period %s - %s' % (date_to_string(period.start),date_to_string(period.stop)))
        return period
    else:
        print('Warning: Did not find valid period, using whole recording period')
        return ceil_periodStart(recording.whole_recording_period)

def ceil_periodStart(period):
    """
    Round up the start of the analysis period to the next second. Unlike NDF, the EDF format does not support miscroseconds
    to be set for "Startdatetime" in the header. This would make the annotation onsets to be misaligned. This isn't a proper 
    solution in case the recording starts at integer value of seconds. But no modulo math needed to calculate exact difference.
    """
    dt = (1e6 - period.start.microsecond) % 1e6
    start = period.start + timedelta(microseconds = dt )
    return nr.nox_recording_class.Period(start, period.stop)

def segment_period(period):
    """
    Split the recording period into smaller segments (1-hour long) defined by tstep variable.
    Necessary to circumvent Nox API limitation when exporting long "Audio" channel that makes it run out of memory.
    Input: period .... an object of "nox_reader.nox_recording_class.Period" type that will be split into segments
    Output: segments .... a list of segments (default: [])
    """
    start = period.start
    tstep = 3600  # Segment size in seconds
    segments = []
    while start < period.stop:
        segment = nr.nox_recording_class.Period(start, start + timedelta(seconds = tstep))
        segments.append(segment)
        start = start + timedelta(seconds = tstep)
    segments[-1] = nr.nox_recording_class.Period(start - timedelta(seconds = tstep), period.stop)  # replace last period with only recorded data
    return segments

def write_signals(recording, period, signal_list, handle):
    """
    Write all signals saved inside NDF "recording" bounded by the "period" into the EDF "file". The waveforms are saved in 16-bit precision.
    The audio waveform is often too big to export in one go so we split it into 1-hour segments.
    Input:
        recording .... an object of "nox_reader.nox_recording_class.Recording" type. Contains everything about the NDF.
        period .... an object of "nox_reader.nox_recording_class.Period" type. Defines the begining and end of exported waveforms.
        handle .... a handle of "pyedflib.edfwriter.EdfWriter" type.
    """
    channel_info = []
    data_list = []
    for signal in signal_list:
        if (signal == 'Audio') and (period.duration > 3600) :
            segments = segment_period(period)
            wave = np.array([], dtype=np.int16)
            for segment in segments:
                chunk = recording.get_signal(signal, segment)
                wave = np.append(wave,np.int16(chunk.data))
            fs = chunk.fs
        else:
            sig = recording.get_signal(signal, period)
            wave = sig.data
            fs = np.int32(np.round(sig.fs))
        label = {'label': signal, 'dimension':'', 'sample_rate': fs, 'physical_max': np.max(wave), 'physical_min': np.min(wave),
                'digital_max': 32767, 'digital_min': -32768, 'transducer': '', 'prefilter':''}
        channel_info.append(label)
        data_list.append(wave)
    handle.setSignalHeaders(channel_info)
    handle.writeSamples(data_list)

def write_annotation_json(recording, period, panel, dir):
    """
    Write all annotations(scorings) from NDF "recording" bounded by the "period" into the JSON file.
    NDF supports more complex annotation structure than EDF so we use JSON. We export only known annotations,
    and in those only valid events.
    Input:
        rec .... an object of "nox_reader.nox_recording_class.Recording" type. Contains everything about the NDF.
        period .... an object of "nox_reader.nox_recording_class.Period" type. Defines the begining and end of exported waveforms.
        handle .... a handle of "pyedflib.edfwriter.EdfWriter" type.  
    """
    scorings = recording.get_all_scoring_names()
    data = {}
    valid_events = ['period-analysisstart','period-analysisstop','snorebreath','breathing-effort']
    for scoring in scorings:
        recording.set_active_scoring_group(scoring,False)
        if scoring == 'ms_snore' or scoring == 'ms_snore_v2':
            events = recording.get_all_events_period(recording.analysis_period, valid_events)
        elif scoring in ['stereo','piezo','mono_m','mono_s','mono_full','cannnula_t3','cannula_a10','stereo_m']:
            events = recording.get_all_events_period(period)
        elif scoring in ['Sigga','PSGPes']:
            valid_events = ['sleep-wake', 'sleep-n1', 'sleep-n2', 'sleep-n3']           
            events = recording.get_all_events_period(recording.analysis_period) 
        else:
            events = recording.get_all_events_period(period)
        data[scoring] = [dict(Event_Type=event.type, Start=date_to_string(event.period.start), 
                            Onset = (event.period.start - period.start).total_seconds(),
                            Duration = event.period.duration) for event in events]
    json_file=os.path.join(dir, panel+'.scoring.json')
    with open(json_file, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, ensure_ascii=False, indent=4)

def write_annotation(recording, period, handle):
    """
    Write all annotations(scorings) saved inside NDF "recording" bounded by the "period" into the EDF file given the "handle".
    NDF supports more complex annotation structure than EDF. Some changes were made to the annotation description inside EDF.
    Read https://www.edfplus.info/specs/edfplus.html#additionalspecs for more info. The output format is 'onset duration description'.
    The onset is the number of seconds since beginning and the duration is in seconds. The description consists of two parts
    <scoring_name>:<event_type>.
    Input:
        rec .... an object of "nox_reader.nox_recording_class.Recording" type. Contains everything about the NDF.
        period .... an object of "nox_reader.nox_recording_class.Period" type. Defines the begining and end of exported waveforms.
        handle .... a handle of "pyedflib.edfwriter.EdfWriter" type.
    """
    scorings = recording.get_all_scoring_names()
    for scoring in ['Sigga']:
        recording.set_active_scoring_group(scoring,False)
        events = recording.get_all_events()
        for event in events:
            if event not in ['sleep-wake', 'sleep-n1', 'sleep-n2', 'sleep-n3']:
                break
            onset = (event.period.start - period.start).total_seconds()
            desc = ':'.join([scoring, event.type])
            handle.writeAnnotation(onset, event.period.duration, desc) 

def write_other(recording, period, panel, handle):
    """
    Write all other available information stored in NDF "recording" bounded by the "period" into the EDF file given the "handle".
    The set specified here is specific to VSN-10-048 dataset. See pyedflib/edfwriter.py for complete list of possible information.
    Important : This has to occure before write_signals or write_annotation!
    Input:
        rec .... an object of "nox_reader.nox_recording_class.Recording" type. Contains everything about the NDF.
        period .... an object of "nox_reader.nox_recording_class.Period" type. Defines the begining and end of exported waveforms.
        handle .... a handle of "pyedflib.edfwriter.EdfWriter" type.
    These fields are also possible to set but I found no reference for them inside recording.
    handle.setTechnician()
    handle.setRecordingAdditional
    handle.setPatientName()
    handle.setPatientAdditional()
    handle.setAdmincode()
    handle.setGender()
    handle.setDatarecordDuration()
    handle.set_number_of_annotation_signals
    handle.setBirthdate()
    """
    handle.setEquipment(recording.get_A1_device_id())
    handle.setStartdatetime(period.start)
    handle.setPatientCode(panel)


## Variable definitions
dataset = r'C:\Users\borsky\Desktop\python\Nox\data'
dir_EDF = r'C:\Users\borsky\Desktop\python\Nox\extract'
periods_file =r'C:\Users\borsky\Desktop\python\Nox\data\signal_periods.txt'
periods = read_period_file(periods_file)


## Main loop
for panel in os.listdir(dataset):
    print(panel)
    recording = os.path.join(dataset, panel)
    if not os.path.isdir(recording):
        continue

    ## Load input recording/panel from the NDF directory
    print('Processing recording: {}'.format(recording))
    rec = nr.get_recording_without_derived(recording)
    period = get_valid_period(rec,panel,periods)
    signal_list = get_valid_signals(rec,period)

    ## Initialize the output EDF file in the EDF directory
    file_EDF = os.path.join(dir_EDF, panel + ".edf")
    handle_EDF = pyedflib.EdfWriter(file_EDF, len(signal_list), file_type=pyedflib.FILETYPE_EDFPLUS)

    ## Write data and close
    print('Saving into file: {}'.format(file_EDF))
    write_other(rec, period, panel, handle_EDF)
    write_signals(rec,period, signal_list, handle_EDF)
    write_annotation_json(rec, period, panel, dir_EDF)
    handle_EDF.close()