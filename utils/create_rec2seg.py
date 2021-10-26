"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.

Not ready for use!
"""
import os
from os import path
import sleepat
from sleepat import io, dsp

def create_rec2seg(data_dir:str, seg_len:float) -> None:
    """
    Creates a rec2seg file that specifies how to segment wave files into non-overlapping
    chunks. The file is saved in data_dir. The script expects wave.scp and annotation to
    make sure the segment doesn't cut an event in half. In order to speed up the search,
    we transform annotation to simple event boundaries. Script expects seg_len, seg_eps
    and wave duration in wave.scp to use the same units (i.e. seconds).
    Input:
        data_dir ... source dir with wave.scp
        seg_len ... segment length (default:float = 10)
        seg_len_min ... minimal segment length (default:float = 1.0)
    """
    print(f'Preparing rec2seg file for {data_dir}.')
    wave_dict = io.read_scp(path.join(data_dir,'wave.scp'))
    seg_len_min = 1

    ## Main
    rec2seg = dict()
    for rec_id, item in wave_dict.items():
        rec_len = dsp.wave_to_len(item['file'], item['fs'])
        (seg_beg,idx) = 0.0, 0
        tmp = dict()

        while (seg_beg+seg_len_min) <= rec_len:
            seg_id = '-'.join([rec_id,'{:04}'.format(idx)])
            seg_end = round(min(seg_beg+seg_len, rec_len),5)
            tmp[seg_id] = {'onset': round(seg_beg,5),'duration': round(seg_end-seg_beg,5)}
            seg_beg = seg_end
            idx += 1
        rec2seg[rec_id] = tmp
    io.write_scp(path.join(data_dir,'rec2seg'),rec2seg)


    """ Legacy code when I checked events before cutting
    annot_dict = io.read_scp(path.join(data_dir,'annotation'))            
            for i, event in enumerate(events):
                event_beg = event['onset']
                event_end = round(event['onset']+event['duration'],5)

                # 0) Event is cleanly before the segment.
                if event_end <= seg_beg:
                    continue
                # 2) Event is cleanly within segment.
                if event_end <= seg_end:
                    continue
                # 3) Event starts within, but ends after segment.
                if event_beg < seg_end:
                    seg_end = event_end
                    if seg_end - seg_beg > max_seg_len:
                        seg_end = max_seg_len
                # 4) Event is cleanly after segment
                if event_beg >= seg_end:
                    events = events[i:]
                    break
                
    """
