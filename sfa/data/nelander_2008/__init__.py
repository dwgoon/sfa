# -*- coding: utf-8 -*-

"""
[Reference]
Nelander, S. et al.
Models from experiments: combinatorial drug perturbations of cancer cells.
Molecular Systems Biology (2008) 4(1), 216.
http://doi.org/10.1038/msb.2008.53
"""

import os

import pandas as pd

import sfa
import sfa.base

def create_data():
    return NelenderData()


class NelenderData(sfa.base.Data):

    def __init__(self):

        self._abbr = "NELANDER_2008"
        self._name = "Nelander et al. 2008 Mol Sys Biol (2008) 4(1), 216"
        inputs = {}
        inputs['EGF'] = 1.0
        self.initialize(__file__, inputs=inputs)
    # end of def __init__


