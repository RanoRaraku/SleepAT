"""
Made by Michal Borsky, 2020, copyright (C) RU.
Evaluate detection accuracy.
"""
import os
from os import path
import sleepat
from sleepat import io, infer, utils


def score_detect(data_dir:str, lang_dir:str, decode_dir:str) -> None:
    """
    Evaluate detection performance which is understood as classification and localization
    problem.
    """
    print(f'Scoring {decode_dir}.')

    # Checks and file loading
    for item in [data_dir,lang_dir,decode_dir]:
        if not path.isdir(item):
            print(f'Error: {item} not found.')
            exit(1)

    # Dump to text files
    res_dir = path.join(decode_dir,f'scoring')
    if not path.isdir(res_dir):
        os.mkdir(res_dir)

    sub2utt = io.read_scp(path.join(data_dir,'spk2utt'))
    utt2sub = io.read_scp(path.join(data_dir,'utt2spk'))
    refer = io.read_scp(path.join(data_dir,'annot'))
    hypth = io.read_scp(path.join(decode_dir,'trans'))
    events = io.read_scp(path.join(lang_dir,'events'))

    # Remove null events
    nonull = [x for x in events.keys() if not (x == events['null']) ]
    for utt in utt2sub.keys():
        refer[utt] = utils.filter_scoring(refer[utt],'label',nonull)
        hypth[utt] = utils.filter_scoring(hypth[utt],'label',nonull)

    for error in ['ser','ier', 'der']:

        # IE is DE with thr = 0, (default thr = 2/3)
        if error == 'ier':
            kwargs = {'thr':0}
            func = getattr(infer,'eval_der')
        else:
            kwargs = {}
            func = getattr(infer,f'eval_{error}')

        # get per utterance/subject/total stats
        per_utt = dict()
        for utt in utt2sub:
            per_utt[utt] = func(refer[utt],hypth[utt],events, **kwargs)
        per_sub = infer.accumulate_score(per_utt, mode='per_sub', sub2utt=sub2utt)
        total = infer.accumulate_score(per_utt, mode='total')

        # Write results to a file
        with open(path.join(res_dir, error),'w') as fid:

            # Per utterance
            print(f'Per Utterance',file=fid)
            for utt, score in per_utt.items():
                res = infer.compute_metrics(score)
                print(f'{utt} - error [%] {res["err"]} - f1 {res["f1"]} - [H/M/FA/C] : {res["score"]}', file=fid)
            print(f'\n',file=fid)

            # Per subject
            print(f'Per Subject',file=fid)            
            for subj, score in per_sub.items():
                res = infer.compute_metrics(score)
                print(f'{subj} - error [%] {res["err"]} - f1 {res["f1"]} - [H/M/FA/C] : {res["score"]}', file=fid)
            print(f'\n',file=fid)            

            # total
            print(f'Total',file=fid)            
            for _, score in total.items():
                res = infer.compute_metrics(score)
                print(f'total - error [%] {res["err"]} - f1 {res["f1"]} - [H/M/FA/C] : {res["score"]}', file=fid)

    io.write_scp(path.join(decode_dir,'annot'),refer)
    print(f'Scoring done.')
