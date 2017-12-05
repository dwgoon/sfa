# -*- coding: utf-8 -*-

"""
[Reference]
[1] Steinway et al.
    Combinatorial interventions inhibit TGFβ-driven epithelial-to-mesenchymal
    transition and support hybrid cellular phenotypes.
    Npj Systems Biology and Applications (2015)  1(1), 15014.


[2] Steinway, S. N. et al.
    Network modeling of TGFβ signaling in hepatocellular carcinoma
    epithelial-to-mesenchymal transition reveals joint sonic hedgehog
    and Wnt pathway activation.
    Cancer Research (2014) 74(21), 5963–77.

[Information]
The topology of network is curated from reference [1].

"""

import os

import pandas as pd

import sfa
import sfa.base


def create_data():
    return SteinwayData()


class SteinwayData(sfa.base.Data):

    def __init__(self):

        self._abbr = "steinway_2015"
        self._name = "Steinway et al. Npj Syst Biol Appl (2015)  1(1), 15014"
        inputs = {}
        inputs['TGFβ'] = 1.0

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

