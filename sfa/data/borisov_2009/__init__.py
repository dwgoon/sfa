# -*- coding: utf-8 -*-

"""
[Reference]
Borisov, N. et al.
Systems-level interactions between insulin-EGF networks amplify mitogenic
signaling.
Molecular Systems Biology (2009) 5(1), 256.
http://doi.org/10.1038/msb.2009.19

[Information]
- The experimental data was generated using ODE model of Borisov et al.
  (It is hypothesized that the ODE model is quite well constructed enough to be
  a substitute for real experimental data).
- The directed network was created by Daewon Lee.
"""

import os
import re

import pandas as pd

import sfa
import sfa.base

p = re.compile("BORISOV_2009_(\S+)_(\S+)")


def create_data(abbr=None):
    if abbr is None:  # Create all data objects
        data_mult = {}  # Multiple data
        list_ba = ["CTRL", "EGF", "I", "EGF+I"]
        for stim_type in list_ba:
            abbr_auc = "BORISOV_2009_AUC_%s"%(stim_type)
            abbr_ss = "BORISOV_2009_SS_%s"%(stim_type)
            data_mult[abbr_auc] = _create_single_data(abbr_auc)
            data_mult[abbr_ss] = _create_single_data(abbr_ss)
        # end of for
        return data_mult
    else: # Create a single data object
        return _create_single_data(abbr)
# end of def


def _create_single_data(abbr):
    dpath = os.path.dirname(__file__)

    m = p.match(abbr)
    try:
        data_type, stim_type = m.groups()
    except AttributeError: # if m is None
        raise ValueError("The wrong abbr. for Borisov 2009 data: %s"%(abbr))

    fstr_ba_file = os.path.join(dpath, "ba.tsv")
    df_ba = pd.read_table(fstr_ba_file,
                          header=0, index_col=0)

    data_type = data_type.lower()
    fstr_exp_file = os.path.join(dpath,
                                 "exp_%s_%s.tsv"%(data_type, stim_type))
    df_exp = pd.read_table(fstr_exp_file,
                           header=0, index_col=0)

    fstr_input_file = os.path.join(dpath, "inputs_%s.tsv" % (stim_type))
    ser_inputs = pd.read_table(fstr_input_file,
                               header=None, index_col=0, squeeze=True)
    inputs = ser_inputs.to_dict()

    return BorisovData(abbr, inputs, df_ba, df_exp)
# end of def


class BorisovData(sfa.base.Data):
    def __init__(self, abbr, inputs, df_ba, df_exp):
        super().__init__()
        self._abbr = abbr
        self._name = "Data generated from Borisov et al. ODE model (%s)"%(abbr)

        dpath = os.path.dirname(__file__)
        fpath = os.path.join(dpath, "network.sif")

        A, n2i, dg = sfa.read_sif(fpath, as_nx=True)
        self._A = A
        self._n2i = n2i
        self._dg = dg
        self._inputs = inputs
        self._df_ba = df_ba
        self._df_exp = df_exp
    # end of def __init__
# end of def class BorisovData