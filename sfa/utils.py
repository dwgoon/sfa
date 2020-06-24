# -*- coding: utf-8 -*-
import sys
if sys.version_info <= (2, 8):
    from builtins import super

import os
import codecs
from collections import defaultdict

import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx


__all__ = ["FrozenClass",
           "Singleton",
           "to_networkx_digraph",
           "normalize",
           "rand_swap",
           "rand_flip",
           "rand_weights",
           "rand_structure",
           "get_akey",
           "get_avalue",]


class FrozenClass(object):

    __isfrozen = False

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError( "%r is a frozen class" % self )
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True

    def _melt(self):
        self.__isfrozen = False

"""
<Reference>
http://stackoverflow.com/questions/3603502/prevent-creating-new-attributes-outside-init
"""
# end of def FrozenClass


def Singleton(_class):
    class __Singleton(_class):
        __instance = None

        def __new__(cls, *args, **kwargs):
            if cls.__instance is None:
                cls.__instance = super().__new__(cls, *args, **kwargs)

                # Creation and initialization of '__initialized'
                cls.__instance.__initialized = False
            # end of if
            return cls.__instance

        def __init__(self, *args, **kwargs):
            if self.__initialized:
                return

            super().__init__(*args, **kwargs)
            self.__initialized = True

        def __repr__(self):
            return '<{0} Singleton object at {1}>'.format(
                _class.__name__, hex(id(self)))

        def __str__(self):
            return super().__str__()
    # end of def class

    __Singleton.__name__ = _class.__name__
    return __Singleton

"""
<References>
http://m.egloos.zum.com/mataeoh/v/7081556
"""
# end of def Singleton

def normalize(A, norm_in=True, norm_out=True):
    # Check whether A is a square matrix
    if A.shape[0] != A.shape[1]:
        raise ValueError(
            "The A (adjacency matrix) should be square matrix.")

    # Build propagation matrix (aka. transition matrix) _W from A
    W = A.copy()

    # Norm. out-degree
    if norm_out == True:
        sum_col_A = np.abs(A).sum(axis=0)
        sum_col_A[sum_col_A == 0] = 1
        if norm_in == False:
            Dc = 1 / sum_col_A
        else:
            Dc = 1 / np.sqrt(sum_col_A)
        # end of else
        W = Dc * W  # This is not matrix multiplication

    # Norm. in-degree
    if norm_in == True:
        sum_row_A = np.abs(A).sum(axis=1)
        sum_row_A[sum_row_A == 0] = 1
        if norm_out == False:
            Dr = 1 / sum_row_A
        else:
            Dr = 1 / np.sqrt(sum_row_A)
        # end of row
        W = np.multiply(W, np.mat(Dr).T)
        # Converting np.mat to ndarray
        # does not cost a lot.
        W = W.A
    # end of if
    """
    The normalization above is the same as the follows:
    >>> np.diag(Dr).dot(A.dot(np.diag(Dc)))
    """
    return W


# end of def normalize
    
def to_networkx_digraph(A, n2i=None):
    if not n2i:
        return nx.from_numpy_array(A, create_using=nx.Digraph)        
    
    i2n = {ix:name for name, ix in n2i.items()}        
    dg = nx.DiGraph()
    ind_row, ind_col = A.to_numpy().nonzero()
    for ix_trg, ix_src in zip(ind_row, ind_col):
        name_src = i2n[ix_src]
        name_trg = i2n[ix_trg]
        sign = np.sign(A[ix_trg, ix_src])
        dg.add_edge(name_src, name_trg)
        dg.edges[name_src, name_trg]['SIGN'] = sign
    # end of for
    return dg
    # end of for
# end of def to_networkx_digraph

def rand_swap(A, nsamp=10, noself=True, pivots=None, inplace=False):
    """Randomly rewire the network connections by swapping.

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix (connection matrix).
    nsamp : int, optional
        Number of sampled connections to rewire
    noself : bool, optional
        Whether to allow self-loop link.
    pivots : list, optional
        Indices of pivot nodes
    inplace : bool, optional
        Modify the given adjacency matrix for rewiring.


    Returns
    -------
    B : numpy.ndarray
        The randomized matrix.
        The reference of the given W is returned, when inplace is True.
    """


    if not inplace:
        A_org = A
        B = A.copy() #np.array(A, dtype=np.float64)
    else:
        A_org = A.copy() #np.array(A, dtype=np.float64)
        B = A

    cnt = 0
    while cnt < nsamp:
        ir, ic = B.to_numpy().nonzero()
        if pivots:
            if np.random.uniform() < 0.5:
                isrc1 = np.random.choice(pivots)
                nz = B[:, isrc1].to_numpy().nonzero()[0]
                if len(nz) == 0:
                    continue
                itrg1 = np.random.choice(nz)
            else:
                itrg1 = np.random.choice(pivots)
                nz = B[itrg1, :].to_numpy().nonzero()[0]
                if len(nz) == 0:
                    continue
                isrc1 = np.random.choice(nz)
            # if-else

            itrg2, isrc2 = itrg1, isrc1
            while isrc1 == isrc2 and itrg1 == itrg2:
                i2 = np.random.randint(0, ir.size)
                itrg2, isrc2 = ir[i2], ic[i2]
        else:
            i1, i2 = 0, 0
            while i1 == i2:
                i1, i2 = np.random.randint(0, ir.size, 2)

            itrg1, isrc1 = ir[i1], ic[i1]
            itrg2, isrc2 = ir[i2], ic[i2]

        if noself:
            if itrg2 == isrc1 or itrg1 == isrc2:
                continue

        # Are the swapped links new?
        if B[itrg2, isrc1] == 0 and B[itrg1, isrc2] == 0:
            a, b = B[itrg1, isrc1], B[itrg2, isrc2]

            # Are the swapped links in the original network?
            if A_org[itrg2, isrc1] == a and A_org[itrg1, isrc2] == b:
                continue

            B[itrg2, isrc1], B[itrg1, isrc2] = a, b
            B[itrg1, isrc1], B[itrg2, isrc2] = 0, 0
            cnt += 1
        else:
            continue
    # end of while

    if not inplace:
        return B


def rand_flip(A, nsamp=10, pivots=None, inplace=False):
    """Randomly flip the signs of connections.

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix (connection matrix).
    nsamp : int, optional
        Number of sampled connections to be flipped.
    pivots : list, optional
        Indices of pivot nodes
    inplace : bool, optional
        Modify the given adjacency matrix for rewiring.

    Returns
    -------
    B : numpy.ndarray
        The randomized matrix.
        The reference of the given W is returned, when inplace is True.
    """
    if not inplace:
        B = A.copy() #np.array(A, dtype=np.float64)
    else:
        B = A

    ir, ic = B.to_numpy().nonzero()
    if pivots:
        iflip = np.random.choice(pivots, nsamp)
    else:
        iflip = np.random.randint(0, ir.size, nsamp)

    B[ir[iflip], ic[iflip]] *= -1
    return B


def rand_weights(W, lb=-3, ub=3, inplace=False):
    """ Randomly sample the weights of connections in W from 10^(lb, ub).

    Parameters
    ----------
    W : numpy.ndarray
        Adjacency (connection) or weight matrix.
    lb : float, optional
        The 10's exponent for lower bound
    inplace : bool, optional
        Modify the given adjacency matrix for rewiring.

    Returns
    -------
    B : numpy.ndarray
        The randomly sampled weight matrix.
        The reference of the given W is returned, when inplace is True.
    """
    if not inplace:
        B = np.array(W, dtype=np.float64)
    else:
        if not np.issubdtype(W.dtype, np.floating):
            raise ValueError("W.dtype given to rand_weights should be "
                             "a float type, not %s"%(W.dtype))

        B = W
    # end of if-else

    ir, ic = B.to_numpy().nonzero()
    weights_rand = 10 ** np.random.uniform(lb, ub,
                                           size=(ir.size,))

    B[ir, ic] = weights_rand*np.sign(B[ir, ic], dtype=np.float)
    """The above code is equal to the following:
    
    for i in range(ir.size):
        p, q = ir[i], ic[i]
        B[p, q] = weights_rand[i] * np.sign(B[p, q], dtype=np.float)
    """
    return B


def rand_structure(A, nswap=10, nflip=10, noself=True, pivots=None, inplace=False):
    if not inplace:
        B = A.copy()
    else:
        B = A
    if nflip > 0:
        B = rand_flip(B, nflip, pivots, inplace)
    if nswap > 0:
        B = rand_swap(B, nswap, noself, pivots, inplace)
    return B


def get_akey(d):
    """Get a key from a given dictionary.
    It returns the first key in d.keys().

    Parameters
    ----------
    d : dict
        Dictionary of objects.

    Returns
    -------
    obj : object
        First item of iter(d.keys()).
    """
    return next(iter(d.keys()))


def get_avalue(d):
    """Get a value from a given dictionary.
    It returns the value designated by sfa.get_akey().

    Parameters
    ----------
    d : dict
        Dictionary of objects.

    Returns
    -------
    obj : object
        First item of d[iter(d.keys())].
    """
    akey = next(iter(d.keys()))
    return d[akey]
