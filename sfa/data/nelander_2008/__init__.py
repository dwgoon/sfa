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

        A, n2i, dg = sfa.read_sif(dpath_network, as_nx=True)
        self._A = A
        self._n2i = n2i
        self._dg = dg
        self._df_ba = pd.read_table(os.path.join(dpath, "ba.tsv"),
                                    header=0, index_col=0)
        self._df_exp = pd.read_table(os.path.join(dpath, "exp.tsv"),
                                     header=0, index_col=0)

        inputs = {}
        inputs['EGF'] = 1.0
        self._inputs = inputs
    # end of def __init__


