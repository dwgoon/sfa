# -*- coding: utf-8 -*-

"""
[Reference]
Fumiã, H. F., & Martins, M. L. 
Boolean Network Model for Cancer Pathways: Predicting 
Carcinogenesis and Targeted Therapy Outcomes.
PLoS ONE, (2013) 8(7), e69008

"""

import os

import pandas as pd

import sfa
import sfa.base


def create_data():
    return FumiaData()


class FumiaData(sfa.base.Data):

    def __init__(self):

        self._abbr = "fumia_2013"
        self._name = "Fumiã et al. PLoS ONE, (2013) 8(7), e69008"
        inputs = {}
        inputs['Mutagen'] = 1.0
        inputs['GFs'] = 1.0
        inputs['Nutrients'] = 1.0
        inputs['TNFα'] = 0.0
        inputs['Hypoxia'] = 1.0

        dpath = os.path.dirname(__file__)
        fpath_network = os.path.join(dpath, 'network.sif')
        A, n2i, dg = sfa.read_sif(fpath_network, as_nx=True)
        self._A = A
        self._n2i = n2i
        self._i2n = {idx: name for name, idx in n2i.items()}
        self._dg = dg
        self._inputs = inputs

        # The following members are not defined due to the lack of data.
        self._df_conds = None
        self._df_exp = None
        self._df_ptb = None
        self._has_link_perturb = False
        self._names_ptb = None
        self._iadj_to_idf = None
    # end of def __init__
# end of def class

