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
- The directed network was created by Daewon Lee (dwl@kaist.ac.kr).
- The units of EGF and insulin (I) stimulation are nM.
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

def create_test_data():
    data_mult = {}
    dpath = os.path.dirname(__file__)

    fstr_file = os.path.join(dpath, 'exp_data_test', 'exp_*')
    for abspath in glob.glob(fstr_file):
        fname = os.path.basename(abspath)
        items = re.split("[_.]", fname)

        data_type = items[1]
        stim_type = items[2]

        if 'EGF' in stim_type:
            conc_EGF = 1
        else:
            conc_EGF = 0.0001

        if 'I' in stim_type:
            conc_I = 1
        else:
            conc_I = 0.0001

        #data_obj = #_create_single_data(abbr, fname=fname)
        #data_mult[data_obj.abbr] = data_obj

        # str_exp_file = os.path.join(dpath, 'exp_data_test', fname)
        # df_exp = pd.read_table(str_exp_file,
        #                        header=0,
        #                        index_col=0)  # Load basal activity data
        #
        # # Load basal activity data
        # str_ba_file = os.path.join(dpath, 'exp_data_test', 'conds.tsv')
        # df_conds = pd.read_table(str_ba_file,
        #                       header=0,
        #                       index_col=0)
        #
        # abbr = "BORISOV_2009_%s_%s"%(data_type, stim_type)
        # data_obj = BorisovData(abbr, data_type, conc_EGF, conc_I, df_conds, df_exp)
        abbr = "BORISOV_2009_%s_%s" % (data_type, stim_type)
        fname_conds = os.path.join('exp_data_test', "conds.tsv")
        fname_exp = os.path.join('exp_data_test', fname)

        data_obj = BorisovData(abbr, data_type,
                               conc_EGF, conc_I,
                               fname_conds, fname_exp)
        data_mult[abbr] = data_obj
    # end of for

    return data_mult



def _create_single_data(abbr=None, fname=None):
    dpath = os.path.dirname(__file__)

    if fname:
        items = re.split('[._+]', fname)

        sim_duration = items[1]
        data_type = items[2]
        stim_EGF = items[3]
        stim_I = items[4]

        m = re.search("EGF=((\d|d)+)", stim_EGF)
        dconc_EGF = m.group(1)
        conc_EGF = dconc_EGF.replace('d', '.')  # Use '.' instead of 'd'

        # Fetch the concentration of I
        m = re.search("I=((\d|d)+)", stim_I)
        dconc_I = m.group(1)
        conc_I = dconc_I.replace('d', '.')  # Use '.' instead of 'd'
        abbr = "%s_%s_EGF=%s+I=%s"%(sim_duration, data_type, conc_EGF, conc_I)

    elif abbr:  # Use abbr
        items = re.split('[_+]', abbr)

        sim_duration = items[1]
        data_type = items[2]
        stim_EGF = items[3]
        stim_I = items[4]

        # Fetch the concentration of EGF
        m = re.search("EGF=((\w|\.)+)", stim_EGF)
        conc_EGF = m.group(1)
        dconc_EGF = conc_EGF.replace('.', 'd')  # Use 'd' instead of '.'

        # Fetch the concentration of I
        m = re.search("I=((\w|\.)+)", stim_I)
        conc_I = m.group(1)
        dconc_I = conc_I.replace('.', 'd')  # Use 'd' instead of '.'

        fname = "exp_%s_%s_EGF=%s+I=%s.tsv" % (sim_duration, data_type,
                                               dconc_EGF, dconc_I)
    else:
        raise ValueError("One of abbr or fname should be given"
                         "in %s._create_single_data()"%(__name__))
    # end of if-else

    fname_exp = os.path.join('exp_data', fname)
    fname_conds = "conds.tsv"

    return BorisovData(abbr, data_type,
                       conc_EGF, conc_I,
                       fname_conds, fname_exp)

    # str_exp_file = os.path.join(dpath, 'exp_data', fname)
    # df_exp = pd.read_table(str_exp_file,
    #                        header=0, index_col=0)
    #
    # # Load basal activity data
    # str_ba_file = os.path.join(dpath, "conds.tsv")
    # df_conds = pd.read_table(str_ba_file,
    #                         header=0,
    #                         index_col=0)

    #return BorisovData(abbr, data_type, conc_EGF, conc_I, df_conds, df_exp)
# end of def

class BorisovData(sfa.base.Data):
    def __init__(self,
                 abbr,
                 data_type,
                 conc_EGF, conc_I,
                 fname_conds, fname_exp):

        super().__init__()
        self._abbr = abbr
        fstr_name = "BORISOV_2009_%s[EGF=%snM,I=%snM]"
        str_name = fstr_name % (data_type, conc_EGF, conc_I)
        self._name = str_name

        inputs = {}
        inputs['EGF'] = 1 #float(conc_EGF)
        inputs['I'] = 1 #float(conc_I)

        self.initialize(__file__,
                        inputs=inputs,
                        fname_conds=fname_conds,
                        fname_exp=fname_exp)

    # end of def __init__
# end of def class BorisovData