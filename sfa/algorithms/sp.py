# -*- coding: utf-8 -*-
"""
@author: dwlee
"""

import numpy as np
import pandas as pd

import sfa.base
from ._propagation import Propagation


def create_algorithm(abbr):
    return SignalPropagation(abbr)
# end of def


class SignalPropagation(Propagation):
    def __init__(self, abbr):
        super().__init__(abbr)        
        self._name = "Signal propagation algorithm"
    # end of def __init__

    def initialize(self, normalize=None):
        super().initialize()
        try:
            P = self._P
            alpha = self._params.alpha
            M0 = np.eye(P.shape[0]) - (1 - alpha) * P
            self._M = alpha * np.linalg.inv(M0)
            self._exsol_avail = True
        except np.linalg.LinAlgError:
            self._exsol_avail = False
    # end of def initialize

# end of def class