
import numpy as np
import scipy as sp
import pandas as pd


__all__ = ["calc_accuracy",]


def calc_accuracy(df1, df2, get_cons=False):
    """
    Count the same sign of each element between df1 and df2

    df1: pandas.DataFrame or numpy.ndarray to be compared
    df2: pandas.DataFrame or numpy.ndarray to be compared
    getcons: decide whether to return consensus array in DataFrame or not
    """

    np.sign(df1) + np.sign(df2)
    if df1.ndim == 1:
        num_total = df1.shape[0]
    elif df1.ndim == 2:
        num_total = df1.shape[0] * df1.shape[1]
    diff_abs = np.abs(np.sign(df1) - np.sign(df2))
    consensus = (diff_abs == 0)

    if isinstance(consensus, pd.DataFrame):
        num_cons = consensus.values.sum()
    else:
        #num_cons = consensus.sum(axis=1).sum()  # Number of consensus
        num_cons = consensus.sum()
        
    acc = (num_cons) / np.float(num_total)  # Accuracy
    if get_cons:
        return acc, consensus
    else:
        return acc


# end of def
