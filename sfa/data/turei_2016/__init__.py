# -*- coding: utf-8 -*-

"""
[Reference]
Türei, D., Korcsmáros, T., & Saez-Rodriguez, J.
OmniPath: Guidelines and gateway for 
literature-curated signaling pathway resources.
Nature Methods, (2016) 13(12), 966–967.
"""

import os

import sfa
import sfa.base


def create_data():
    return TureiData()


class TureiData(sfa.base.Data):

    def __init__(self):

        self._abbr = "turei_2016"
        self._name = "Türei et al. " \
                     "Nature Methods, (2016) 13(12), 966–967"
        inputs = {}

        dpath = os.path.dirname(__file__)
        fpath_network = os.path.join(dpath, 'network_giant_component.sif')
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