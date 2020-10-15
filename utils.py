from matplotlib.pyplot import axis
import numpy as np
from collections import Counter
from numpy.lib.function_base import average
import sklearn


def filenames():
    '''This func returns a dict of filenames based on the alphabet (english |
    greek) and the set that you want (train | test | etc.) so that it can be
    accessed dynamically'''

    f = {
        'english': {
            'letters': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
            'train': './data/Assig1-Dataset/train_1.csv',
            'test': './data/Assig1-Dataset/test_with_label_1.csv',
            'val': './data/Assig1-Dataset/val_1.csv',
            'nolabel': './data/Assig1-Dataset/test_no_label_1.csv'
        },
        'greek': {
            'letters': ['π', 'α', 'β', 'σ', 'γ', 'δ', 'λ', 'ω', 'µ', 'ξ'],
            'train': './data/Assig1-Dataset/train_2.csv',
            'test': './data/Assig1-Dataset/test_with_label_2.csv',
            'val': './data/Assig1-Dataset/val_2.csv',
            'nolabel': './data/Assig1-Dataset/test_no_label_2.csv'
        }
    }

    return f


def load_data(filename):
    '''
    Loads data from the filename provided and returns X (Nx1024) and y (Nx1)
    '''
    X = np.loadtxt(filename, dtype=np.float64, delimiter=',', skiprows=1)  # NX1025
    y = X[:, -1]  # the last column contains the labels Nx1
    X = X[:, :-1]  # Nx1024

    return X, y


def compute_metrics(model, X, y,):
    '''
    Computes the confusion matrix, class metrics, f1 macro & weighted averages.
    Outputs an array, a list of class metrics & each f1 score individually.
    '''

    y_pred = model.predict(X)
    confMatrix = sklearn.metrics.confusion_matrix(y, y_pred)
    bin_matrix, row_sums = compute_individual_metrics(confusion_matrix=confMatrix)
    macrof1, weightf1 = compute_model_fscore(bin_matrix, row_sums)

    return confMatrix, bin_matrix, macrof1, weightf1


def compute_individual_metrics(confusion_matrix):
    '''
    Input: sklearn.metrics.confusion_matrix (rows = true, columns = predicted)
    Output: a list of dicts. List index i contains a dict of TP, TN, FP, FN,
    precision, recall, f1 for label i.
    '''

    true_row_sum = np.sum(confusion_matrix, dtype=np.int32, axis=1)
    pred_col_sum = np.sum(confusion_matrix, dtype=np.int32, axis=0)
    total = np.sum(confusion_matrix, dtype=np.int32)
    binary_conf_matrix = []

    for label in range(len(confusion_matrix)):
        tp = confusion_matrix[label, label]
        fn = true_row_sum[label] - tp
        fp = pred_col_sum[label] - tp
        tn = total - fn - fp + tp
        precision = tp/(tp+fp) if tp+fp != 0 else 0
        recall = tp/(tp+fn) if tp+fn != 0 else 0
        binary_conf_matrix.append({'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
                                   'precision': precision,
                                   'recall': recall,
                                   'f1': 2*precision*recall/(precision+recall) if precision+recall != 0 else 0})

    return binary_conf_matrix, true_row_sum


def compute_model_fscore(binary_conf_matrix, true_row_sum):
    '''
    Input: from the compute_individual_metrics func
    Output: returns the model macro f1 and weighted f1
    '''
    macro = 0
    weighted = 0
    for i in range(len(binary_conf_matrix)):
        x = binary_conf_matrix[i]
        macro += x['f1']
        weighted += x['f1'] * true_row_sum[i]

    total_instances = np.sum(true_row_sum)
    macro = macro / len(binary_conf_matrix)
    weighted = weighted / total_instances

    return macro, weighted
