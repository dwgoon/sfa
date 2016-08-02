# -*- coding: utf-8 -*-

"""
[Reference]
Borisov, N. et al. (2009).
Systems-level interactions between insulin-EGF networks amplify mitogenic
signaling. Molecular Systems Biology, 5(1), 256.
http://doi.org/10.1038/msb.2009.19

[Information]
- The experimental data was generated using ODE model of Borisov et al.
  (It is hypothesized that the ODE model is quite well constructed enough to be
  a substitute for real experimental data).
- The directed network was created by Daewon Lee.
"""

import os

import pandas as pd

import sfa
import sfa.base


def create_data():
    data_mult = {} # Multiple data

    list_ba = ["lowstim", "EGF", "I", "EGF+I"]
    dpath = os.path.dirname(__file__)
    for cond in list_ba:
        df_ba = pd.read_table("%s/ba_%s.tsv"%(dpath, cond),
                              header=0, index_col=0)
        df_exp_auc = pd.read_table("%s/exp_auc_%s.tsv"%(dpath, cond),
                                   header=0, index_col=0)
        df_exp_ss = pd.read_table("%s/exp_ss_%s.tsv"%(dpath, cond),
                                  header=0, index_col=0)

        key_auc = "BORISOV_AUC_%s"%(cond)
        key_ss = "BORISOV_SS_%s"%(cond)

        data_auc = BorisovData(key_auc, df_ba, df_exp_auc)
        data_ss = BorisovData(key_ss, df_ba, df_exp_ss)

        data_mult[key_auc] = data_auc
        data_mult[key_ss] = data_ss
    # end of for

    return data_mult


# end of def

class BorisovData(sfa.base.Data):
    def __init__(self, abbr, df_ba, df_exp):
        super().__init__(abbr)
        self._name = "Data generated from Borisov et al. ODE model (%s)"%(abbr)

        dpath = os.path.dirname(__file__)
        fpath = os.path.join(dpath, "network.sif")

        A, n2i = sfa.read_sif(fpath)
        self._A = A
        self._n2i = n2i
        self._dg = sfa.read_sif(fpath, as_nx=True)
        self._df_ba = df_ba
        self._df_exp = df_exp
    # end of def __init__
# end of def class BorisovData