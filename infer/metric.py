"""
Made by Michal Borsky, 2020, copyright (C) RU.

A set of evaluation measures for classification
tasks. All rely on a confusion matrix to exist.
"""
import numpy as np

def conf_matrix(pred:np.ndarray,tgt:np.ndarray, classes:dict) -> np.ndarray:
    """
    Creates a confusion matrix (n,n) where x-axis is target
    and y-axis is prediction. The values are raw numbers, by
    default. 

    Arguments:
        pred ... prediction array as numerical values or strings
        target ... target array as numerical values or string
        classes ... dictionary with class labels and numerical values
        <normalize> ... normalize matrix values to unity (defualt:bool = False)
    Return:
        cm ... a confusion matrix
    Use:
        infer.conf_matrix(pred,target,)
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
    return cm

def accuracy(conf_matrix):
    """
    Calculates an accuracy of classification, defined as
    sum of diagonal over sum of all elements. Confusion
    matrix can be of arbitrary (n,n) dimensionality. Rows
    are targets/actual class and columne are prediction.
    See infer.conf_matric() for more details.

    Arguments:
        conf_matrix ... confusion matrix of (n,n) dimensions
    Return:
        accuracy ... a scalar in 0-1 range
    Use:
        acc = sleepat.infer.accuracy(conf_matrix)
    """
    return conf_matrix.trace()/conf_matrix.sum() 


def precision(label, confusion_matrix):
    """
    """
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()
    
def recall(label, confusion_matrix):
    """
    """
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()


def f1_score(confusion_matrix):
    """
    """
    return 1

def auc(confusion_matrix):
    """
    """
    return 1