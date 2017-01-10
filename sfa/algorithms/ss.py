# -*- coding: utf-8 -*-
import sys
if sys.version_info <= (2, 8):
    from builtins import super


import numpy as np
from ._np import NetworkPropagation


def create_algorithm(abbr):
    return SignalSmoothingWithoutNorm(abbr)
# end of def


class SignalSmoothingWithoutNorm(NetworkPropagation):

    class ParameterSet(NetworkPropagation.ParameterSet):
        def initialize(self):
            super().initialize()
            self._apply_weight_norm = False

    # end of class ParameterSet


    def __init__(self, abbr):
        super().__init__(abbr)
        self._name = "Signal smoothing algorithm without normalization"

    def _prepare_exact_solution(self):
        a = self._params.alpha
        W = self._W
        Dw = np.diag(np.diag(W))
        Wn = W - Dw
        Wn_abs = np.abs(Wn)
        n = W.shape[0]

        # The effect of self-loop
        Ds = ((np.eye(n) - np.sign(Dw)) ** 2).dot(np.abs(Dw))
        # Sum. of in-links effects
        S_in = np.diag(Wn_abs.sum(axis=1))
        # Sum. of out-links effects
        S_out = np.diag(Wn_abs.sum(axis=0))

        M0 = 0.5*a*((Ds+S_in+S_out) - (Wn+Wn.T)) + (1-a)*np.eye(n)

        if np.linalg.det(M0) == 0:
            raise np.linalg.LinAlgError()

        self._M = np.linalg.inv(M0) * (1-a)
        self._weight_matrix_invalidated = False
    # end of def _prepare_exact_solution

    def _prepare_iterative_solution(self):
        # To get Dc and W_ot
        W = self.W
        a = self._params.alpha
        diag_W = np.diag(W)
        Dw = np.diag(diag_W)
        Wn = W - Dw
        Wn_abs = np.abs(Wn)

        # The effect of self-loop
        Ds = ((1-np.sign(diag_W))**2)*np.abs(diag_W)

        # Sum. of in-links effects
        S_in = Wn_abs.sum(axis=1)

        # Sum. of out-links effects
        S_out = Wn_abs.sum(axis=0)

        # Coefficient matrix
        self._Dc = np.diag(1.0/(0.5*a*(Ds+S_in+S_out) + (1-a)))

        """
        The above calculation is the same as the following:
        (the above calculation does not use the inversion,
         which can be simply achieved by division of every element)

        Ds = ((np.eye(n) - np.sign(Dw)) ** 2).dot(np.abs(Dw))
        S_in = np.diag(Wn_abs.sum(axis=1))
        S_out = np.diag(Wn_abs.sum(axis=0))
        Dc = np.linalg.inv(a*(Ds+S_in+S_out) + (1-a)*np.eye(n))
        """

        # W(original + transposed) = Wn + Wn.T
        self._W_ot = 0.5*(Wn + Wn.T)
        self._weight_matrix_invalidated = False
    # end of def _prepare_iterative_solution

    def propagate_exact(self, b):
        if self._weight_matrix_invalidated:
            self._prepare_exact_solution()

        return self._M.dot(b)

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
            Propagation terminates, when the iteration is reached.
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

        """
        The Dc and W_ot matrices are created or changed
        in the function, self._prepare_iterative_solution(), depending on W.
        """

        if self._weight_matrix_invalidated:
            self._prepare_iterative_solution()

        Dc = self._Dc
        W_ot = self._W_ot

        # Main loop
        num_iter = 0
        for i in range(lim_iter):
            # Main formula
            x_t2 = Dc.dot(a*W_ot.dot(x_t1) + (1-a) * b)
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
# end of def class SignalSmoothing
