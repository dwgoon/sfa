# -*- coding: utf-8 -*-

import sfa

from .sp import SignalPropagation


def create_algorithm(abbr):
    return GaussianSmoothing(abbr)
# end of def


class GaussianSmoothing(SignalPropagation):
    def __init__(self, abbr):
        super().__init__(abbr)
        self._name = "Gaussian smoothing algorithm"       

    def _normalize(self, A):
        return A  # No norm.

    def _prepare_exact_solution(self):
        pass

    def propagate_exact(self, b):
        pass

    def propagate_iterative(self,
                            P,
                            xi,
                            b,
                            a=0.5,
                            lim_iter=1000,
                            tol=1e-5,
                            notrj=True):
        pass

