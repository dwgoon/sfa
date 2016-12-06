# -*- coding: utf-8 -*-

import sys
if sys.version_info <= (2, 8):
    from builtins import super

import numpy as np
import pandas as pd


import sfa.base
import sfa.utils

from .sp import SignalPropagation
from .aps import AcyclicPathSummation


def create_algorithm(abbr):
    return CyclicPathSummation(abbr)
# end of def


class CyclicPathSummation(SignalPropagation):

    class ParameterSet(SignalPropagation.ParameterSet):
        """
        Parameters of CyclicPathSummation algorithm.
        """
        def initialize(self):
            super().initialize()
            self._lim_iter = 1000
            self._exsol_forbidden = True
            self._apply_weight_norm = False

    # end of def class ParameterSet

    def __init__(self, abbr):
        super().__init__(abbr)
        self._name = "Cyclic path summation algorithm"
        self._params = self.ParameterSet()
    # end of def __init__

    def _prepare_exact_solution(self):
        raise np.linalg.LinAlgError()
    # end of def _prepare_exact_solution

    def _prepare_iterative_solution(self):
        self.params.exsol_forbidden = True
        self._exsol_avail = False
        self._weight_matrix_invalidated = False
    # end of def _prepare_iterative_solution

    def compute(self, b):
        alpha = self._params.alpha
        W = self.W
        lim_iter = self._params.lim_iter
        x_sum, _ = self.propagate_iterative(W, b, a=alpha,
                                            lim_iter=lim_iter)
        return x_sum
    # end of def compute

    def propagate_exact(self, b):
        raise NotImplementedError()

    def propagate_iterative(self,
                            W,
                            b,
                            a=0.5,
                            lim_iter=1000,
                            tol=1e-8,
                            get_trj=False):

        x_sum = np.zeros_like(b)
        #x_sum += b  # Add the starting point (i.e., length-zero path).
        x_t1 = b
        trj_x = []

        # Main loop
        num_iter = 0
        for i in range(lim_iter):
            x_t2 = a*W.dot(x_t1)
            x_sum += x_t2
            num_iter += 1

            # Check termination condition
            if np.linalg.norm(x_t2 - x_t1) <= tol:
                break

            # Add the current state to the trajectory
            if get_trj:
                trj_x.append(x_t2)

            # Update the state
            x_t1 = x_t2.copy()
        # end of for

        if get_trj is False:
            return x_sum, num_iter
        else:
            return x_sum, np.array(trj_x)

    # end of def propagate_iterative


# end of def class CyclicPathSummation
