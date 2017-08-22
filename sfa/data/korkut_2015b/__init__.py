# -*- coding: utf-8 -*-

"""
[Reference]


[Information]

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

from .. import korkut_2015a


def create_data():
    data_mult = {}  # Multiple data
    dpath = os.path.dirname(__file__)

    fstr_file = os.path.join(dpath, 'networks', '*.sif')
    for abspath in glob.glob(fstr_file):
        fname = os.path.basename(abspath)
        data_obj = _create_single_data(fname=fname)
        data_mult[data_obj.abbr] = data_obj

    # end of for

    return data_mult

# end of def


def _create_single_data(fname):
    dpath = os.path.dirname(korkut_2015a.__file__)
    fpath_network = os.path.join(os.path.dirname(__file__),
                                 'networks',
                                 fname)    
    abbr = fname.split('.')[0].upper()
    return korkut_2015a.KorkutData(abbr, dpath, fpath_network)
#    if fname:
#        items = re.split('[._+]', fname)
#
#        sim_duration = items[1]
#        data_type = items[2]
#        stim_EGF = items[3]
#        stim_I = items[4]
#
#        m = re.search("EGF=((\d|d)+)", stim_EGF)
#        dconc_EGF = m.group(1)
#        conc_EGF = dconc_EGF.replace('d', '.')  # Use '.' instead of 'd'
#
#        # Fetch the concentration of I
#        m = re.search("I=((\d|d)+)", stim_I)
#        dconc_I = m.group(1)
#        conc_I = dconc_I.replace('d', '.')  # Use '.' instead of 'd'
#        abbr = "%s_%s_EGF=%s+I=%s"%(sim_duration, data_type, conc_EGF, conc_I)
#
#    elif abbr:  # Use abbr
#        items = re.split('[_+]', abbr)
#
#        sim_duration = items[1]
#        data_type = items[2]
#        stim_EGF = items[3]
#        stim_I = items[4]
#
#        # Fetch the concentration of EGF
#        m = re.search("EGF=((\w|\.)+)", stim_EGF)
#        conc_EGF = m.group(1)
#        dconc_EGF = conc_EGF.replace('.', 'd')  # Use 'd' instead of '.'
#
#        # Fetch the concentration of I
#        m = re.search("I=((\w|\.)+)", stim_I)
#        conc_I = m.group(1)
#        dconc_I = conc_I.replace('.', 'd')  # Use 'd' instead of '.'
#
#        fname = "exp_%s_%s_EGF=%s+I=%s.tsv" % (sim_duration, data_type,
#                                               dconc_EGF, dconc_I)
#    else:
#        raise ValueError("One of abbr or fname should be given"
#                         "in %s._create_single_data()"%(__name__))
#    # end of if-else
#
#    fname_exp = os.path.join('exp_data', fname)
#    fname_conds = "conds.tsv"

#    return KorkutData(abbr, data_type,
#                       conc_EGF, conc_I,
#                       fname_conds, fname_exp)
# end of def

class KorkutData(sfa.base.Data):
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

