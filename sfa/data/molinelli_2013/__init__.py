# -*- coding: utf-8 -*-

"""
[Reference]
Molinelli, E. J. et al.
Perturbation biology: inferring signaling networks in cellular systems.
PLoS Computational Biology, (2013) 9(12), e1003290.
http://doi.org/10.1371/journal.pcbi.1003290

[Information]

"""

import os

import pandas as pd

import sfa
import sfa.base

def create_data():
    return MolinelliData()


class MolinelliData(sfa.base.Data):

    def __init__(self):

        self._abbr = "MOLINELLI_2013"
        self._name = "Molinell et al. 2013 PLoS Comput Biol 9(12): e1003290"

        sfa.create_data_members(self, __file__)

#        dpath = os.path.dirname(__file__)
#        dpath_network = os.path.join(dpath, "network.sif")
#
#        A, n2i, dg = sfa.read_sif(dpath_network, as_nx=True)
#        self._A = A
#        self._n2i = n2i
#        self._dg = dg
#        self._df_conds = pd.read_table(os.path.join(dpath, "conds.tsv"),
#                                       header=0, index_col=0)
#        self._df_exp = pd.read_table(os.path.join(dpath, "exp_sub.tsv"),
#                                     header=0, index_col=0)

    # end of def __init__




