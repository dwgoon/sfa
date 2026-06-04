# -*- coding: utf-8 -*-

import numpy as np

from .np import NetworkPropagation
from .np import NetworkPropagationParameterSet


def create_algorithm(abbr):
    return SignalPropagation(abbr)
# end of def


class SignalPropagationParameterSet(NetworkPropagationParameterSet):
    def initialize(self):
        super().initialize()
# end of class ParameterSet


class SignalPropagation(NetworkPropagation):

    def __init__(self, abbr):
        super().__init__(abbr)
        self._name = "Signal propagation algorithm"
    # end of def __init__

    def prepare_exact_solution(self):
        r"""
        Prepare the matrix $M$ for the closed-form steady-state solution.

        Starting from

        $$
        x(t+1) = \alpha W x(t) + (1 - \alpha) b,
        $$

        as $t \to \infty$ both $x(t+1)$ and $x(t)$ converge to the
        stationary state $s$, giving

        $$
        s = \alpha W s + (1 - \alpha) b
          \implies (I - \alpha W) s = (1 - \alpha) b
          \implies s = M b,
          \quad M = (1 - \alpha)(I - \alpha W)^{-1}.
        $$

        This method computes and caches $M$.
        """
        W = self._W
        a = self._params.alpha
        M0 = np.eye(W.shape[0]) - a*W
        self._M = (1-a)*np.linalg.inv(M0)
    # end of def _prepare_exact_solution

    def prepare_iterative_solution(self):
        pass  # Nothing...
    # end of def prepare_iterative_solution

    def propagate_exact(self, b):
        if self._weight_matrix_invalidated:
            self.prepare_exact_solution()
            
        return self._M.dot(b)
    # end of def propagate_exact

    def propagate_iterative(self,
                            W,
                            xi,
                            b,
                            a=0.5,
                            lim_iter=1000,
                            tol=1e-5,
                            get_trj=False):


        n = W.shape[0]
        # Initial values
        x0 = np.array(xi, dtype=np.float64)

        x_t1 = x0.copy()

        if get_trj:
            # Record the initial states
            trj_x = []
            trj_x.append(x_t1.copy())

        # Main loop
        num_iter = 0
        for i in range(lim_iter):
            # Main formula
            x_t2 = a*W.dot(x_t1) + (1-a)*b
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
            return x_t2, num_iter
        else:
            return x_t2, np.array(trj_x)

    # end of def propagate_iterative
# end of def class SignalPropagation
