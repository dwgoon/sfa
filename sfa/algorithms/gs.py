# -*- coding: utf-8 -*-

import sfa

from ._propagation import Propagation

def create_algorithm(abbr):
    return GaussianSmoothing(abbr)
# end of def


class GaussianSmoothing(Propagation):
    def __init__(self, abbr):
        super().__init__(abbr)
        self._name = "Gaussian smoothing algorithm"       

    def _propagate(self, b):
        if self._exsol_avail:
            return self._M.dot(b)
        else:
            alpha = self._params.alpha
            P = self._P
            x_ss, _ = self.propagate(P, b, b, a=alpha)
            return x_ss  # x at steady-state (i.e., staionary state)

    # end of def _propagate

    def compute(self):
        print("Computing Gaussian smoothing ...")