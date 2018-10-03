# -*- coding: utf-8 -*-

"""
[Reference]
Cho et al.
Attractor landscape analysis of 
colorectal tumorigenesis and its reversion.
BMC Systems Biology (2016) 10:96.
"""

import os

import pandas as pd

import sfa
import sfa.base


def create_data():
    return ChoData()


class ChoData(sfa.base.Data):

    def __init__(self):

        self._abbr = "CHO_2015"
        self._name = "Cho et al. BMC Systems Biology (2016) 10:96"
        
        inputs = {}
        inputs['ECM'] = 1.0
        inputs['Tgf-b'] = 1.0
        inputs['IL1-TNF'] = 1.0
        inputs['EGF'] = 1.0
        inputs['alpha_i_lig'] = 1.0
        inputs['alpha_12_13_lig'] = 1.0
        inputs['alpha_s_lig'] = 1.0
        inputs['alpha_q_lig'] = 1.0
        inputs['Stress'] = 1.0
        inputs['WNT'] = 1.0
        inputs['Fas'] = 1.0
        inputs['ExtPump'] = 1.0
        inputs['DNA_damage'] = 1.0

        dpath = os.path.dirname(__file__)
        fpath_network = os.path.join(dpath, 'network_all_pos.sif')
        A, n2i, dg = sfa.read_sif(fpath_network, as_nx=True)
        self._A = A
        self._n2i = n2i
        self._i2n = {idx: name for name, idx in n2i.items()}
        self._dg = dg
        self._inputs = inputs


    # end of def __init__
# end of def class

