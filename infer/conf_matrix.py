"""
Made by Michal Borsky, 2020, copyright (C) RU
"""
import numpy as np

def conf_matrix(pred:np.ndarray,tgt:np.ndarray, classes:dict,
    normalize:bool=False) -> np.ndarray:
    """
    Creates a confusion matrix (n,n) where x-axis is target
    and y-axis is prediction.

    Arguments:
        pred ... prediction array as numerical values or strings
        target ... target array as numerical values or string
        classes ... dictionary with class labels and numerical values
        <normalize> ... normalize matrix values to unity (defualt:bool = False)
    Return
        cm ... confusion matrix
    """
    if not classes:
        print(f'Classes dictionary is empty.')
        exit(1)

    classes_num = len(set(list(classes.values())))
    if classes_num < 1:
        print(f'Error: number of classes < 2.')
        exit(1)

    cm = np.zeros((classes_num,classes_num), dtype=np.int64)
    for i,j in zip(pred,tgt):
        cm[i,j] += 1

    if normalize:
        cm /= cm.sum()
    return cm
