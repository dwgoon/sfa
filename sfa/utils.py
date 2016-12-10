# -*- coding: utf-8 -*-
import sys
if sys.version_info <= (2, 8):
    from builtins import super

import os
import codecs
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx


import sfa


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
#
# def create_data_members(self,
#                         fpath,
#                         fname_network="network.sif",
#                         fname_ptb="ptb.tsv",
#                         fname_conds="conds.tsv",
#                         fname_exp="exp.tsv",
#                         inputs={}):
#
#     dpath = os.path.dirname(fpath)
#     fpath_network = os.path.join(dpath, fname_network)
#     fpath_ptb = os.path.join(dpath, fname_ptb)
#
#     A, n2i, dg = sfa.read_sif(fpath_network, as_nx=True)
#     self._A = A
#     self._n2i = n2i
#     self._dg = dg
#     self._df_conds = pd.read_table(os.path.join(dpath, fname_conds),
#                                    header=0, index_col=0)
#     self._df_exp = pd.read_table(os.path.join(dpath, fname_exp),
#                                  header=0, index_col=0)
#
#     self._inputs = inputs
#     self._df_ptb = pd.read_table(fpath_ptb, index_col=0)
#     if any(self._df_ptb.Type == 'link'):
#         self._has_link_perturb = True
#     else:
#         self._has_link_perturb = False
#
#     self._names_ptb = []
#     for i, row in enumerate(self._df_conds.iterrows()):
#         row = row[1]
#         list_name = []  # Target names
#         for target in self._df_conds.columns[row.nonzero()]:
#             list_name.append(target)
#         # end of for
#         self._names_ptb.append(list_name)
#     # end of for
#
#     # For mapping from the indices of adj. matrix to those of DataFrame
#     # (arrange the indices of adj. matrix according to df_exp.columns)
#     self._iadj_to_idf = [n2i[x] for x in self._df_exp.columns]
#
#     self._i2n = {idx: name for name, idx in n2i.items()}
# # end of def

def normalize(A, norm_in=True, norm_out=True):
    # Check whether A is a square matrix
    if A.shape[0] != A.shape[1]:
        raise ValueError(
            "The A (adjacency matrix) should be square matrix.")

    # Build propagation matrix (aka. transition matrix) W from A
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

def calc_accuracy(df1, df2, get_cons=False):
    """
    Count the same sign of each element between df1 and df2

    df1: Left pandas.DataFrame to be compared
    df2: Right pandas.DataFrame to be compared
    getcons: decide whether to return consensus array in DataFrame or not
    """

    np.sign(df1) + np.sign(df2)
    num_total = df1.shape[0]*df1.shape[1]
    diff_abs = np.abs(np.sign(df1) - np.sign(df2))    
    consensus = (diff_abs == 0)
    
    num_cons = consensus.sum(axis=1).sum()  # Number of consensus
    acc = (num_cons) / np.float(num_total)  # Accuracy
    if get_cons:
        return acc, consensus
    else:
        return acc
# end of def


def get_akey(d):
    """
    Get a key from a given dictionary.
    It returns the first key in d.keys().
    """
    return next( iter(d.keys()) )


def get_avalue(d):
    """
    Get a value from a given dictionary.
    It returns the value designated by sfa.get_akey().
    """
    akey = next( iter(d.keys()) )
    return d[akey]