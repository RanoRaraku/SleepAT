"""
Made by Michal Borsky, 2020, copyright (C) RU.
Compute standard performance measures.
"""
import numpy as np

def compute_metrics(score:np.ndarray) -> dict:
    """
    Compute standard performance measures of multiclass detection.

    precision = (Hit) / (Hit + Miss + Conf)
    recall = (Hit) / ((Hit + FA + Conf))
    F1 = 2* precision*recall / (precision + recall)
    error = (Miss + FA + Conf) / (Hit + Miss + Conf)

    Arguments:
        score ...

    Return:
        {'score':[H/M/FA/C], 'error': error}
    """
    # Checks
    (h,m,fa,c) = score

    prec = h/(h+c+m)
    rec = h/(h+c+fa)
    f1 = round( 2*(prec*rec) / (prec+rec), 4)
    err = round((c+m+fa)/(h+c+m),4)

    return({'precision':round(prec,4), 'recall':round(rec,4), 'F1':f1, 'error':err})
