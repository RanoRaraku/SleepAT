"""
Made by Michal Borsky, 2019, copyright (C) RU

Collection of utility routines to manipulate datasets, do checks.
Some functions are generators and have return in loop.
"""
import sleepat
from sleepat import io

def merge_scp(*args,scp_out:str) -> None:
    """
    Merges a number of scp files into one scp. The values in merged
    scp are stored in a list. Different keys can have different no.
    values. The script is used to merge multiple feature scps without
    merging the data files, which is usually be done on the fly.
    Other use is to rename foo.scp to feats.scp, other scripts expect it.
    Input:
        args ... a list of input scp files
        scp_out .... name of output scp file
    """
    if len(args) < 1:
        print('utils.merge_scp(): No input scp files were defined.')
        exit(1)

    if len(args) ==1:
        io.write_scp(scp_out, io.read_scp(args[0]))
        return

    merged = dict()
    for file in args:
        scp = io.read_scp(file)
        for key,item in scp.items():
            if key in merged:
                merged[key].append(item)
            else:
                merged[key] = [item] 
    io.write_scp(scp_out, merged)
