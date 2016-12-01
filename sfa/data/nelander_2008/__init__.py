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

        dpath = os.path.dirname(__file__)
        dpath_network = os.path.join(dpath, "network.sif")
        dpath_ptb = os.path.join(dpath, "ptb.tsv")

        A, n2i, dg = sfa.read_sif(dpath_network, as_nx=True)
        self._A = A
        self._n2i = n2i
        self._dg = dg
        self._df_conds = pd.read_table(os.path.join(dpath, "conds.tsv"),
                                       header=0, index_col=0)
        self._df_exp = pd.read_table(os.path.join(dpath, "exp.tsv"),
                                     header=0, index_col=0)

        inputs = {}
        inputs['EGF'] = 1.0
        self._inputs = inputs
        self._df_ptb = pd.read_table(dpath_ptb, index_col=0)

        
        # N = self._data.A.shape[0]  # Number of state variables
        #df_conds = self._data.df_conds  # Basal activity
        #n2i = self.data.n2i

        self._names_ptb = []
        self._vals_ptb = []
        self._types_ptb = []
        for i, row in enumerate(self._df_conds.iterrows()):
            row = row[1]
            list_name = []  # Target names
            list_val = []  # Values
            list_type = []  # Perturbation types
            for target in self._df_conds.columns[row.nonzero()]:
                list_name.append(target)
                list_val.append(self._df_ptb.ix[target, 'Value'])
                list_type.append(self._df_ptb.ix[target, 'Type'])
            # end of for
            self._names_ptb.append(list_name)
            self._vals_ptb.append(list_val)
            self._types_ptb.append(list_type)
        # end of for

        # For mapping from the indices of adj. matrix to those of DataFrame
        # (arrange the indices of adj. matrix according to df_exp.columns)
        self._iadj_to_idf = [n2i[x] for x in self._df_exp.columns]

        self._i2n = {idx: name for name, idx in n2i.items()}
    # end of def __init__


