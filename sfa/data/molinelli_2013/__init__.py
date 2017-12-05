# -*- coding: utf-8 -*-

"""
[Reference]
Molinelli, E. J. et al.
Perturbation biology: inferring signaling networks in cellular systems.
PLoS Computational Biology, (2013) 9(12), e1003290.
http://doi.org/10.1371/journal.pcbi.1003290

"""

import os

import pandas as pd

import sfa
import sfa.base

def create_data():
    return MolinelliData()


class MolinelliData(sfa.base.Data):

    def __init__(self):

        self._abbr = "MOLINELLI_2013"
        self._name = "Molinell et al. 2013 PLoS Comput Biol 9(12): e1003290"

        self.initialize(__file__)

    # end of def __init__




