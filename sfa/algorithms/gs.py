# -*- coding: utf-8 -*-
import sys
if sys.version_info <= (2, 8):
    from builtins import super

import numpy as np

import sfa
from .sp import SignalPropagation


def create_algorithm(abbr):
    return GaussianSmoothing(abbr)
# end of def


class GaussianSmoothing(SignalPropagation):
    def __init__(self, abbr):
        super().__init__(abbr)
        self._name = "Gaussian smoothing algorithm"       

    def normalize(self, A):
        return A  # No norm.

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
        D_in = np.diag(Wn_abs.sum(axis=1))
        # Sum. of out-links effects
        D_out = np.diag(Wn_abs.sum(axis=0))

        M0 = a * ((Ds + D_in + D_out) - (Wn + Wn.T)) + (1-a) * np.eye(n)
        if np.linalg.det(M0) == 0:
            raise np.linalg.LinAlgError()

        return np.linalg.inv(M0) * (1-a)

    def propagate_exact(self, b):
        return self._M.dot(b)

    def propagate_iterative(self,
                            W,
                            xi,
                            b,
                            a=0.5,
                            lim_iter=1000,
                            tol=1e-5,
                            trj=False):
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
        gettrj: bool (optional)
            Determine whether trajectory of the state and propagation matrix
            is returned. If gettrj is true, the trajectory is returned.

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

        if gettrj:
            # Record the initial states
            trj_x = []
            trj_x.append(x_t1.copy())

        # To get Dc and W_ot
        Dw = np.diag(np.diag(W))
        Wn = W - Dw
        Wn_abs = np.abs(Wn)
        n = W.shape[0]

        # The effect of self-loop
        Ds = ((np.eye(n) - np.sign(Dw)) ** 2).dot(np.abs(Dw))
        # Sum. of in-links effects
        D_in = np.diag(Wn_abs.sum(axis=1))
        # Sum. of out-links effects
        D_out = np.diag(Wn_abs.sum(axis=0))

        # Coefficient matrix
        Dc = np.linalg.inv(a*(Ds + D_in + D_out) + (1-a)*np.eye(n))

        # W(original + transposed) = Wn + Wn.T
        W_ot = Wn + Wn.T

        # Main loop
        num_iter = 0
        for i in range(lim_iter):
            # Main formula
            x_t2 = Dc.dot(a * W_ot.dot(x_t1) + (1 - a) * b)
            num_iter += 1
            # Check termination condition
            if np.linalg.norm(x_t2 - x_t1) <= tol:
                break

            # Add the current state to the trajectory
            if gettrj:
                trj_x.append(x_t2)

            # Update the state
            x_t1 = x_t2.copy()
        # end of for

        if gettrj is False:
            return x_t2, num_iter
        else:
            return x_t2, np.array(trj_x)
    # end of def propagate_iterative
# end of def class GaussianSmoothing
