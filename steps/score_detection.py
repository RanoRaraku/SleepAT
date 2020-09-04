"""
Made by Michal Borsky, 2020, copyright (C) RU.
Evaluate detection accuracy.
"""
import os
from os import path
import sleepat
from sleepat import io, infer, utils

def score_detection(data_dir:str, lang_dir:str, decode_dir:str) -> None:
    """
    Evaluate detection performance which is understood as classification and localization
    problem.

    """
    # Checks and file loading
    for item in [data_dir,lang_dir,decode_dir]:
        if not path.isdir(item):
            print(f'Error: {item} not found.')
            exit(1)
    utt2spk = io.read_scp(path.join(data_dir,'utt2spk'))
    annot = io.read_scp(path.join(data_dir,'annot'))
    periods = io.read_scp(path.join(data_dir,'periods'))
    targets = io.read_scp(path.join(data_dir,'targets.scp'))
    trans = io.read_scp(path.join(decode_dir,'trans'))
    post = io.read_scp(path.join(decode_dir,'post.scp'))
    events = io.read_scp(path.join(lang_dir,'events'))

    # Compute Frame Error Rate (FER)
    fer = infer.compute_fer(targets,post,events)

    # Compute Detection Error Rate
    # Normalize annotation/transcription and generate alignment
    align = dict()
    for utt_id in utt2spk.keys():
        annot[utt_id] = utils.normalize_scoring(annot[utt_id],periods[utt_id],events)
        trans[utt_id] = utils.normalize_scoring(trans[utt_id],periods[utt_id],events)
        align[utt_id] = infer.align_scorings(annot[utt_id],trans[utt_id])
    (align,score)= infer.compute_der(annot,trans,align,events)

    # Dump results
    msg = (f"score_detection() --data_dir={data_dir} --lang_dir={lang_dir} --decode_dir={decode_dir}\n\n"\
        f"%FER {fer['FER']} [ {fer['tot']} tot / {fer['cor']} cor]\n" \
        f"%DER {der['DER']} [ {sum(der['ISDC'])} tot / {der['ISDC'][0]} ins, {der['ISDC'][1]} sub, "\
        f"{der['ISDC'][2]} del, {der['ISDC'][3]} cor]\n"\
        f"MAD(event_start) = {der['Beg_MAD']}, MAD(event_end) = {der['End_MAD']}\n"\
        f"Scored {len(align)} utterances."\
    )
    with open(path.join(decode_dir,'score'), 'w') as fid:
        print(msg,file=fid)

    # Dump per utterance alingnment
    msg = ('')
    for utt_id,ali in align.items():
        (score,isdc) = '', ''
        ref_txt = [i['label'] for i in annot[utt_id]]
        hyp_txt = [i['label'] for i in trans[utt_id]]

        for pair in ali:
            score = " ".join([score, str(pair['Score'])])
            isdc = " ".join([isdc, pair['ISDC']])
        msg += f'{utt_id} Ref  : {ref_txt}\n'
        msg += f'{utt_id} Hyp  : {hyp_txt}\n'
        msg += f'{utt_id} Score: {score}\n'
        msg += f'{utt_id} ISDC : {isdc}\n'
    with open(path.join(decode_dir,'details'), 'w') as fid:
        print(msg,file=fid)

    # Dump annot for easier reference
    io.write_scp(path.join(decode_dir,'annot'),annot)

