# -*- coding: utf-8 -*-

import sys
if sys.version_info <= (2, 8):
    from builtins import super

import numpy as np

import sfa.base
import sfa.utils

from ._np import NetworkPropagation

def create_algorithm(abbr):
    return SignalPropagation(abbr)
# end of def


class SignalPropagation(NetworkPropagation):

    class ParameterSet(NetworkPropagation.ParameterSet):
        def initialize(self):
            super().initialize()
    # end of class ParameterSet


    def __init__(self, abbr):
        super().__init__(abbr)
        self._name = "Signal propagation algorithm"
    # end of def __init__


    def _prepare_exact_solution(self):
        """
        Prepare to get the matrix for the exact solution:

        x(t+1) = a*W.dot(x(t)) + (1-a)*b, where a is alpha.

        When t -> inf, both x(t+1) and x(t) converges to the stationary state.

        Then, s = aW*s + (1-a)b
              (I-aW)*s = (1-a)b
              s = (I-aW)^-1 * (1-a)b
              s = M*b, where M is (1-a)(I-aW)^-1.

        This method is to get the matrix, M for preparing the exact solution
        """
        W = self._W
        a = self._params.alpha
        M0 = np.eye(W.shape[0]) - a*W
        self._M = (1-a)*np.linalg.inv(M0)
    # end of def _prepare_exact_solution

    def _prepare_iterative_solution(self):
        pass
    # end of def _prepare_iterative_solution

    def propagate_exact(self, b):
        if self._weight_matrix_invalidated:
            self._prepare_exact_solution()
            
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
        """
        Network propagation calculation based on iteration.

        Parameters
        ------------------
        W: numpy.ndarray
           weight matrix (transition matrix)
        xi: numpy.ndarray
            Initial state
        b: numpy.ndarray
            Basal activity
        a: real number (optional)
            Propagation rate.
        lim_iter: integer (optioanl)
            Limitation of iterations for propagation.
            Propagation terminates, when the iterat

            ion is reached.
        tol: real number (optional)
            Tolerance for terminating iteration
            Iteration continues, if Frobenius norm of (x(t+1)-x(t)) is
            greater than tol.
        get_trj: bool (optional)
            Determine whether trajectory of the state and propagation matrix
            is returned. If get_trj is true, the trajectory is returned.

        Returns
        -------
        xp: numpy.ndarray
            State after propagation
        trj_x: numpy.ndarray
            Trajectory of the state transition

        See also
        --------
        """

        n = W.shape[0]
        # Initial values
        x0 = np.zeros((n,), dtype=np.float)
        x0[:] = xi

        x_t1 = x0.copy()

        if get_trj:
            # Record the initial states
            trj_x = []
            trj_x.append(x_t1.copy())

        # Main loop
        num_iter = 0
        for i in range(lim_iter):
            # Main formula
            x_t2 = a * W.dot(x_t1) + (1 - a) * b
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
