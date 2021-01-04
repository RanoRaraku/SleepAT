"""
Made by Michal Borsky, 2020, copyright (C) RU.
Compute standard performance measures.
"""
import numpy as np

def compute_metrics(score:np.ndarray) -> dict:
    """
    Compute standard performance measures of multiclass detection. The input is a 
    numpy array of hits, misses, false alarms, confusion in this order. All values
    are non-negative and at least 1 item is > 0.

    precision = Hit / (Hit + Miss + Conf)
    recall = Hit / (Hit + FA + Conf)
    F1 = 2* precision*recall / (precision + recall)
    error = (Miss + FA + Conf) / (Hit + Miss + Conf)
    jaccard index = Hit / (Hit + Miss + Conf + FA)
    jaccard distance = 1 - JI = (Hit + Miss + Conf) / (Hit + Miss + Conf + FA)

    Arguments:
        score ... a list/tuple/numpy array of (h,m,fa,c) in this order
    Return:
        {'score':[H/M/FA/C], 'error': error}
    """
    # Checks
    if not isinstance(score,np.ndarray):
        print(f'Error compute_metrics(): score is not a numpy array.')
        exit(1)
    if score.any() < 0:
        print(f'Error compute_metrics(): some values are negative {(score)}.')
        exit(1)
    if not score.any():
        print(f'Error compute_metrics(): score is all zeros {(score)}.')
        exit(1)


    # Calculate performance metrics
    (h,m,fa,c) = score
    prec = h/(h+c+m)
    rec = h/(h+c+fa)
    f1 = round(2*(prec*rec)/(prec+rec), 4)
    err = round(100*(m+fa+c)/(h+c+m), 2)
    ji = round(h/(h+c+m+fa),4)
    jd = round(1 - ji,4)

    return {'score':score, 'prec':round(prec,4), 'rec':round(rec,4),
            'f1':f1, 'err':err, 'ji':ji, 'jd':jd}
