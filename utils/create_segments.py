"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
from os.path import join
from sleepat.io import read_scp
from sleepat.dsp import wave_to_len

def create_segments(data_dir:str, segm_len:float, segm_len_min:float) -> None:
    """
    Creates a segments file that specifies how to segment wave files into non-overlapping chunks.
    The file is saved in data_dir. The script expects wave.scp and annotation to make sure the
    segment doesn't cut an event in half. In order to speed up the search, we transform annotation
    to simple event boundaries. Script expects seg_len, seg_eps and wave duration in wave.scp to
    use the same units (i.e. seconds). The segments file is saved into data_dir/segments.
    Input:
        data_dir .... source dir with wave.scp and annotation and where segments is saved
        seg_len .... tentative segment length (default:float = 10)
        min_seg_len .... minimal segment length (default:float = 1)
        max_seg_len .... maximum segments length (default:float = 15)
    """
    print('Preparing segments file for %s' % data_dir)

    ## Config section
    annot_dict = read_scp(join(data_dir,'annotation'))
    wave_dict = read_scp(join(data_dir,'wave.scp'))

    ## Main
    segments = dict()
    for utt_id, item in wave_dict.items():
        utt_len = wave_to_len(item['file'], item['fs'])
        (seg_beg,idx) = 0.0, 0
        tmp = dict()

        while seg_beg < utt_len:
            # Setup a preliminary segment end
            seg_id = '-'.join([utt_id,'{:04}'.format(idx)])
            seg_end = round(min(seg_beg+segm_len, utt_len),5)
            if seg_end + segm_len_min >= utt_len:
                seg_end = utt_len
            """ Legacy code when I checked events before cutting
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
            tmp[seg_id] = {'onset': round(seg_beg,5),'duration': round(seg_end-seg_beg,5)}
            seg_beg = seg_end
            idx += 1
        segments[utt_id] = tmp
    return segments
