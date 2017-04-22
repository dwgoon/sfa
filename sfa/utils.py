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
           "read_sif",
           "normalize",
           "randswap",
           "randflip",
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


def read_sif(filename, sym_pos='+', sort=True, as_nx=False):
    dict_links = defaultdict(list)
    set_nodes = set()
    name_to_idx = {}
    with codecs.open(filename, "r", encoding="utf-8") as f_in:
        for line in f_in:
            items =  line.strip().split()
            src = items[0]
            tgt = items[2]
            sign = items[1]

            set_nodes.add( src )
            set_nodes.add( tgt )
            if sign == sym_pos:
                dict_links[src].append( (tgt, 1) )
            else:
                dict_links[src].append( (tgt, -1) )
        # end of for
    # end of with

    if sort == True:
        list_nodes = sorted(set_nodes)
    else:
        list_nodes = list(set_nodes)

    N = len(set_nodes)
    adj = np.zeros((N, N), dtype=np.int)

    for isrc, name in enumerate(list_nodes):
        name_to_idx[name] = isrc  # index of source
    # end of for
    for name_src in name_to_idx:
        isrc = name_to_idx[name_src]
        for name_tgt, sign in dict_links[name_src]:
            itgt = name_to_idx[name_tgt]
            adj[itgt, isrc] = sign
            # end of for
    # end of for

    if not as_nx:
        return adj, name_to_idx
    else: # NetworkX DiGraph
        dg = nx.DiGraph()
        # Add nodes
        for name in list_nodes:
            dg.add_node(name)
            
        # Add edges (links)
        for name_src in list_nodes:
            for name_tgt, sign in dict_links[name_src]:
                dg.add_edge(name_src, name_tgt,
                            attr_dict={'sign': sign})
            # end of for
        # end of for
        return adj, name_to_idx, dg
    # end of else
# end of def


def normalize(A, norm_in=True, norm_out=True):
    # Check whether A is a square matrix
    if A.shape[0] != A.shape[1]:
        raise ValueError(
            "The A (adjacency matrix) should be square matrix.")

    # Build propagation matrix (aka. transition matrix) _W from A
    W = A.copy()

    # Norm. in-degree
    if norm_in == True:
        sum_col_A = np.abs(A).sum(axis=0)
        sum_col_A[sum_col_A == 0] = 1
        if norm_out == False:
            Dc = 1 / sum_col_A
        else:
            Dc = 1 / np.sqrt(sum_col_A)
        # end of else
        W = Dc * W  # This is not matrix multiplication

    # Norm. out-degree
    if norm_out == True:
        sum_row_A = np.abs(A).sum(axis=1)
        sum_row_A[sum_row_A == 0] = 1
        if norm_in == False:
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

def randswap(A, nsamp=10, noself=True, inplace=False):
    """Randomly rewire the network connections by swapping.

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix (connection matrix).
    nsamp : int, optional
        Number of sampled connections to rewire
    noself : bool, optional
        Whether to allow self-loop link.
    inplace : bool, optional
        Modify the given adjacency matrix for rewiring.


    Returns
    -------
    B : numpy.ndarray
        New adjacency matrix.
        None is return when inplace is True.
    """


    if not inplace:
        A_org = A
        B = A.copy()
    else:
        A_org = A.copy()
        B = A

    cnt = 0
    while cnt < nsamp:
        ir, ic = B.nonzero()
        i1, i2 = np.random.randint(0, ir.size, 2)

        itgt1, isrc1 = ir[i1], ic[i1]
        itgt2, isrc2 = ir[i2], ic[i2]

        if noself:
            if itgt2 == isrc1 or itgt1 == isrc2:
                continue

        if B[itgt2, isrc1] == 0 and B[itgt1, isrc2] == 0:
            a, b = B[itgt1, isrc1], B[itgt2, isrc2]
            if A_org[itgt2, isrc1] == a and A_org[itgt1, isrc2] == b:
                continue

            B[itgt2, isrc1], B[itgt1, isrc2] = a, b
            B[itgt1, isrc1], B[itgt2, isrc2] = 0, 0
            cnt += 1
        else:
            continue
    # end of while

    if not inplace:
        return B


def randflip(A, nsamp=10, inplace=False):
    """Randomly flip the signs of connections.

    Parameters
    ----------
    A : numpy.ndarray
        Adjacency matrix (connection matrix).
    nsamp : int, optional
        Number of sampled connections to be flipped.
    inplace : bool, optional
        Modify the given adjacency matrix for rewiring.

    Returns
    -------
    B : numpy.ndarray
        New adjacency matrix.
        None is return when inplace is True.
    """
    if not inplace:
        B = A.copy()
    else:
        B = A

    ir, ic = B.nonzero()
    iflip = np.random.randint(0, ir.size, nsamp)
    B[ir[iflip], ic[iflip]] *= -1

    if not inplace:
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