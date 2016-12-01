# -*- coding: utf-8 -*-

"""
[Reference]
Dalle Pezze, P. et al.
A Dynamic Network Model of mTOR Signaling Reveals
TSC-Independent mTORC2 Regulation.
Science Signaling. 2012 Mar 27;5(217):ra25
http://doi:10.1126/scisignal.2002469

[Information]
- The experimental data was generated using ODE model of Pezze et al.
  (It is hypothesized that the ODE model is quite well constructed enough to be
  a substitute for real experimental data).
- The directed network was created by Junsoo Kang (reality312@kaist.ac.kr).
- The unit of insulin (I) stimulation is nM.
"""

import sys
if sys.version_info <= (2, 8):
    from builtins import super

import os
import re
import glob

import pandas as pd

import sfa
import sfa.base


def create_data(abbr=None):
    if abbr is None:  # Create all data objects
        data_mult = {}  # Multiple data
        dpath = os.path.dirname(__file__)

        fstr_file = os.path.join(dpath, 'exp_data', 'exp_*')
        for abspath in glob.glob(fstr_file):
            fname = os.path.basename(abspath)
            data_obj = _create_single_data(abbr, fname=fname)
            data_mult[data_obj.abbr] = data_obj

        # end of for

        return data_mult
    else:  # Create a single data object
        return _create_single_data(abbr)

# end of def

def _create_single_data(abbr=None, fname=None):
    dpath = os.path.dirname(__file__)

    if fname:
        items = re.split('[._]', fname)

        sim_duration = items[1]
        data_type = items[2]
        stim_I = items[3]  # The concentration of insulin stimulation

        # Fetch the concentration of I
        m = re.search("I=((\d|d)+)", stim_I)
        dconc_I = m.group(1)
        conc_I = dconc_I.replace('d', '.')  # Use '.' instead of 'd'
        abbr = "%s_%s_I=%s"%(sim_duration, data_type, conc_I)

    elif abbr:  # Use abbr
        items = re.split('[_]', abbr)

        sim_duration = items[1]
        data_type = items[2]
        stim_I = items[3]  # The concentration of insulin stimulation

        # Fetch the concentration of I
        m = re.search("I=((\w|\.)+)", stim_I)
        conc_I = m.group(1)
        fname = "exp_%s_%s_I=%s.tsv" % (sim_duration, data_type, conc_I)
    else:
        raise ValueError("One of abbr or fname should be given"
                         "in %s._create_single_data()"%(__name__))
    # end of if-else

    str_exp_file = os.path.join(dpath, 'exp_data', fname)
    df_exp = pd.read_table(str_exp_file,
                           header=0, index_col=0)

    # Load basal activity data
    str_ba_file = os.path.join(dpath, "ba.tsv")
    df_conds = pd.read_table(str_ba_file,
                          header=0, index_col=0)

    return PezzeData(abbr, data_type, conc_I, df_conds, df_exp)
# end of def


class PezzeData(sfa.base.Data):
    def __init__(self, abbr, data_type, conc_I, df_conds, df_exp):
        super().__init__()
        self._abbr = abbr

        dpath = os.path.dirname(__file__)
        fpath = os.path.join(dpath, "network.sif")

        A, n2i, dg = sfa.read_sif(fpath, as_nx=True)
        self._A = A
        self._n2i = n2i
        self._dg = dg
        self._df_conds = df_conds
        self._df_exp = df_exp

        inputs = {}
        inputs['I/IGF'] = float(conc_I)
        self._inputs = inputs

        fstr_name = "PEZZE_2011_%s[I=%snM]"
        str_name = fstr_name % (data_type, conc_I)
        self._name = str_name

    # end of def __init__
# end of def class