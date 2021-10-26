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
    print(f'Scoring {data_dir} vs .{decode_dir}.')

    # Checks and file loading
    for item in [data_dir,lang_dir,decode_dir]:
        if not path.isdir(item):
            print(f'Error: {item} not found.')
            exit(1)

    # Dump to text files
    res_dir = path.join(decode_dir,f'scoring')
    if not path.isdir(res_dir):
        os.mkdir(res_dir)

    sub2rec = io.read_scp(path.join(data_dir,'sub2rec'))
    rec2sub = io.read_scp(path.join(data_dir,'rec2sub'))
    ref = io.read_scp(path.join(data_dir,'annot'))
    hyp = io.read_scp(path.join(decode_dir,'annot'))
    events = io.read_scp(path.join(lang_dir,'events'))

    # Remove null events
    nonull = [x for x in events.keys() if not (x == events['null']) ]
    for rec in rec2sub.keys():
        ref[rec] = utils.filter_scoring(ref[rec],'label',nonull)
        hyp[rec] = utils.filter_scoring(hyp[rec],'label',nonull)

    for error in ['ser','ier', 'der','ssde0','ssde05']:
   # for error in ['ssde0','ssde05']:
        # IE is DE with thr = 0, (default thr = 2/3)

        if error == 'ier':
            kwargs = {'thr':0}
            func = getattr(infer,'eval_der')
        elif error == 'ssde0':
            kwargs = {'thr':0}
            func = getattr(infer,'eval_ssde')
        elif error == 'ssde05':
            kwargs = {'thr':2/3}
            func = getattr(infer,'eval_ssde')   
        else:
            kwargs = {}
            func = getattr(infer,f'eval_{error}')

        # get per recerance/subject/total stats
        per_rec = dict()
        for rec in rec2sub:
            per_rec[rec] = func(ref[rec],hyp[rec],events, **kwargs)
        per_sub = infer.accumulate_score(per_rec, mode='per_sub', sub2rec=sub2rec)
        total = infer.accumulate_score(per_rec, mode='total')

        # Write results to a file
        with open(path.join(res_dir, error),'w') as fid:

            # Per recording
            print(f'Per recording',file=fid)
            for rec, score in per_rec.items():
                res = infer.compute_metrics(score)
                print(f'{rec} - error [%] {res["err"]} - f1 {res["f1"]} - [H/M/FA/C] : {res["score"]}', file=fid)
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

    print(f'Done, results in {decode_dir}/scoring.')
