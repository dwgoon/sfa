# -*- coding: utf-8 -*-

"""
[Reference]
Zañudo, J. G. T., & Albert, R.
Cell Fate Reprogramming by Control of Intracellular Network Dynamics.
PLoS Computational Biology, (2015) 11(4), e1004193.

"""

import os

import pandas as pd

import sfa
import sfa.base


def create_data():
    return ZanudoAData()


class ZanudoAData(sfa.base.Data):

    def __init__(self):

        self._abbr = "zanudo_2015a"
        self._name = "Zañudo et al. PLoS Computational Biology, 11(4), e1004193"
        inputs = {}

        dpath = os.path.dirname(__file__)
        fpath_network = os.path.join(dpath, 'network.sif')
        A, n2i, dg = sfa.read_sif(fpath_network, as_nx=True)
        self._A = A
        self._n2i = n2i
        self._i2n = {idx: name for name, idx in n2i.items()}
        self._dg = dg
        self._inputs = inputs

        # The following members are not defined due to the lack of data.
        self._df_conds = None
        self._df_exp = None
        self._df_ptb = None
        self._has_link_perturb = False
        self._names_ptb = None
        self._iadj_to_idf = None
    # end of def __init__
# end of def class

