"""
Made by Michal Borsky, 2019, copyright (C) RU
Basic IO routines.
"""
import json

def write_scp(file:str, content:dict=None) -> None:
    """
    Write content into a json file. The content is a dictionary where the 'key' is utt_id.
    Indexing by utt_id allows to match contents between multiple json_files. This is used
    to create audio.scp, feats.scp and target.scp and many others. Content cannot contain
    duplicate values so utt_id have to be unique .
    Audio.scp maps between utt_id and audio files used to extract the features.
    Feats.scp maps between utt_id and files that contain extracted features.
    Segments maps between utt_it and segments that will be cut out prior to feature extraction.
    Annotation maps between utt_id and all scored events used to crate target.scp.
    Target.scp maps between utt_it and files that contain targets for each feature file. 
    Input:
        file .... path to the output file
        content .... a dictionary of the format dict[utt_id] = value (default:dict = None)
    """
    if not isinstance(file,str):
        print('Wrong input type, expected string.')
        exit(1)

    if not isinstance(content,dict):
        print('Wrong input type, expected dict.')
        exit(1)

    with open(file, 'w', encoding = 'utf-8') as fh:
        json.dump(content,fh, ensure_ascii=False, indent=1)