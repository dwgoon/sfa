# -*- coding: utf-8 -*-

"""
[Reference]
Nelander, S. et al.
Models from experiments: combinatorial drug perturbations of cancer cells.
Molecular Systems Biology (2008) 4(1), 216.
http://doi.org/10.1038/msb.2008.53

[Information]

"""

import os

import pandas as pd

import sfa
import sfa.base

def create_data():
    return NelenderData()


class NelenderData(sfa.base.Data):

    def __init__(self):

        self._abbr = "NELANDER_2008"
        self._name = "Nelander et al. 2008 Mol Sys Biol (2008) 4(1), 216"
        inputs = {}
        inputs['EGF'] = 1.0

        sfa.create_data_members(self, __file__, inputs=inputs)

        # dpath = os.path.dirname(__file__)
        # fpath_network = os.path.join(dpath, "network.sif")
        # fpath_ptb = os.path.join(dpath, "ptb.tsv")
        #
        # A, n2i, dg = sfa.read_sif(fpath_network, as_nx=True)
        # self._A = A
        # self._n2i = n2i
        # self._dg = dg
        # self._df_conds = pd.read_table(os.path.join(dpath, "conds.tsv"),
        #                                header=0, index_col=0)
        # self._df_exp = pd.read_table(os.path.join(dpath, "exp.tsv"),
        #                              header=0, index_col=0)
        #
        # inputs = {}
        # inputs['EGF'] = 1.0
        # self._inputs = inputs
        # self._df_ptb = pd.read_table(fpath_ptb, index_col=0)
        #
        # if any(self._df_ptb.Type == 'link'):
        #     self._has_link_perturb = True
        # else:
        #     self._has_link_perturb = False
        #
        # self._names_ptb = []
        # for i, row in enumerate(self._df_conds.iterrows()):
        #     row = row[1]
        #     list_name = []  # Target names
        #     for target in self._df_conds.columns[row.nonzero()]:
        #         list_name.append(target)
        #     # end of for
        #     self._names_ptb.append(list_name)
        # # end of for
        #
        # # For mapping from the indices of adj. matrix to those of DataFrame
        # # (arrange the indices of adj. matrix according to df_exp.columns)
        # self._iadj_to_idf = [n2i[x] for x in self._df_exp.columns]
        #
        # self._i2n = {idx: name for name, idx in n2i.items()}
    # end of def __init__


