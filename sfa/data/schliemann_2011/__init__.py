# -*- coding: utf-8 -*-

"""
[Reference]
Schliemann, M. et al.
Heterogeneity reduces sensitivity of cell death for TNF-Stimuli.
BMC Systems Biology 2011, 5:204
http://www.biomedcentral.com/1752-0509/5/204

[Information]
- The experimental data was generated using ODE model of Schliemann et al.
  (It is hypothesized that the ODE model is quite well constructed enough to be
  a substitute for real experimental data).
- The directed network was created by Junsoo Kang (reality312@kaist.ac.kr).
- The unit of TNF stimulation is nM.
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
        m = re.search("TNF=((\d|d)+)", stim_I)
        dconc_I = m.group(1)
        conc_I = dconc_I.replace('d', '.')  # Use '.' instead of 'd'
        abbr = "%s_%s_TNF=%s"%(sim_duration, data_type, conc_I)

    elif abbr:  # Use abbr
        items = re.split('[_]', abbr)

        sim_duration = items[1]
        data_type = items[2]
        stim_I = items[3]  # The concentration of insulin stimulation

        # Fetch the concentration of I
        m = re.search("TNF=((\w|\.)+)", stim_I)
        dconc_I = m.group(1)
        conc_I = dconc_I.replace('d', '.')  # Use '.' instead of 'd'
        fname = "exp_%s_%s_I=%s.tsv" % (sim_duration, data_type, conc_I)
    else:
        raise ValueError("One of abbr or fname should be given"
                         "in %s._create_single_data()"%(__name__))
    # end of if-else

    fname_exp = os.path.join('exp_data', fname)
    fname_conds = "conds.tsv"

    return SchliemannData(abbr, data_type,
                          conc_I,
                          fname_conds, fname_exp)


# end of def


class SchliemannData(sfa.base.Data):
    def __init__(self,
                 abbr,
                 data_type,
                 conc_I,
                 fname_conds, fname_exp):

        super().__init__()
        self._abbr = abbr

        fstr_name = "SCHLIEMANN_2011_%s[I=%snM]"
        str_name = fstr_name % (data_type, conc_I)
        self._name = str_name

        inputs = {}
        inputs['TNF'] = 1 #float(conc_I)
        self._inputs = inputs

        self.initialize(__file__,
                        inputs=inputs,
                        fname_conds=fname_conds,
                        fname_exp=fname_exp)
    # end of def __init__
# end of def class