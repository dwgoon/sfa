# -*- coding: utf-8 -*-

"""
[Reference]


[Information]

"""

import os

import pandas as pd

import sfa
import sfa.base

def create_data():
    return KorkutData()


class KorkutData(sfa.base.Data):

    def __init__(self):

        self._abbr = "KORKUT_2015"
        self._name = "Korkut and Wang et al. eLife 2015;4:e04640"

        #sfa.create_data_members(self, __file__,
        #                        fname_network="average_network.sif")

        dpath = os.path.dirname(__file__)
        fpath_network = os.path.join(dpath, "average_network.sif")
        fpath_ptb = os.path.join(dpath, "ptb.tsv")

        A, n2i, dg = sfa.read_sif(fpath_network, as_nx=True)
        self._A = A
        self._n2i = n2i
        self._dg = dg
        self._df_conds = pd.read_table(os.path.join(dpath, "conds.tsv"),
                                       header=0, index_col=0)
        self._df_exp = pd.read_table(os.path.join(dpath, "exp.tsv"),
                                     header=0, index_col=0)

        self._inputs = {}
        self._df_ptb = pd.read_table(fpath_ptb, index_col=0)
        if any(self._df_ptb.Type == 'link'):
            self._has_link_perturb = True
        else:
            self._has_link_perturb = False

        self._names_ptb = []
        for i, row in enumerate(self._df_conds.iterrows()):
            row = row[1]
            list_name = []  # Target names
            for target in self._df_conds.columns[row.nonzero()]:
                list_name.append(target)
            # end of for
            self._names_ptb.append(list_name)
        # end of for

        s1 = set(self._df_exp.columns)  # From experimental data
        s2 = set(n2i.keys())  # From network
        common = s1 - s2
        self._df_exp.drop(common, axis=1, inplace=True)

        # For mapping from the indices of adj. matrix to those of DataFrame
        # (arrange the indices of adj. matrix according to df_exp.columns)
        self._iadj_to_idf = [n2i[x] for x in self._df_exp.columns]
        #self._iadj_to_idf = [n2i[x] for x in self._n2i]

        self._i2n = {idx: name for name, idx in n2i.items()}
    # end of def __init__




