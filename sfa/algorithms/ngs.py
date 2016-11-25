# -*- coding: utf-8 -*-
import sys
if sys.version_info <= (2, 8):
    from builtins import super


from functools import reduce
import numpy as np

import sfa
from .gs import GaussianSmoothing


def create_algorithm(abbr):
    return NormalizedGaussianSmoothing(abbr)
# end of def


class NormalizedGaussianSmoothing(GaussianSmoothing):

    class ParameterSet(GaussianSmoothing.ParameterSet):
        def initialize(self):
            super().initialize()
    # end of class ParameterSet

    def __init__(self, abbr):
        super().__init__(abbr)
        self._name = "Normalized gaussian smoothing algorithm"

    def _prepare_solution_common(self):
        """
        Perform the calculation the common parts
        in preparing the exact and iterative solutions.
        """

        W = self._W
        diag_W = np.diag(W)
        Dw = np.diag(diag_W)
        Wn = W - Dw
        n = W.shape[0]

        Wn_abs = np.abs(Wn)

        D_in = Wn_abs.sum(axis=1)
        D_out = Wn_abs.sum(axis=0)

        sqrt_D_in = np.sqrt(D_in)
        sqrt_D_out = np.sqrt(D_out)

        pad_sqrt_D_in = sqrt_D_in.copy()
        pad_sqrt_D_out = sqrt_D_out.copy()

        pad_sqrt_D_in[sqrt_D_in == 0] = 1
        pad_sqrt_D_out[sqrt_D_out == 0] = 1


        # The effect of self-loop
        D_tot = D_in + D_out
        inv_D_tot = D_tot.copy()
        inv_D_tot[D_tot == 0] = 1.0
        inv_D_tot = 1.0 / inv_D_tot #np.diag(1.0 / inv_D_tot)

        #Ds = reduce(np.dot, [(np.eye(n) - np.sign(Dw)) ** 2,
        #                     np.abs(Dw),
        #                     inv_D_tot])

        # Element-wise multiplication of three vectors (numpy.ndarray)
        Ds = ((1 - np.sign(diag_W))**2 * np.abs(diag_W) * inv_D_tot)

        # Sum. of in-links effects
        S_in = np.sign(D_in)

        # Sum. of out-links effects
        S_out = np.sign(D_out)

        """
        Normalization of weight matrix
        Inverse of square root of in-degree and out-degree (ISD)
        i.e.,) ISD_in  = (sqrt(in-degree))^-1
               ISD_out = (sqrt(out-degree))^-1
        """
        ISD_in = np.diag(1.0 / pad_sqrt_D_in)
        ISD_out = np.diag(1.0 / pad_sqrt_D_out)

        Wn_norm = reduce(np.dot, [ISD_in, Wn, ISD_out])

        return Ds, S_in, S_out, Wn_norm
    # end of def

    def _prepare_exact_solution(self):
        a = self._params.alpha
        [Ds, S_in, S_out, Wn_norm] = self._prepare_solution_common()
        Ds = np.diag(Ds)
        S_in = np.diag(S_in)
        S_out = np.diag(S_out)

        n = Wn_norm.shape[0]
        M0 = a*((Ds+S_in+S_out) - (Wn_norm+Wn_norm.T)) + (1-a)*np.eye(n)

        if np.linalg.det(M0) == 0:
            raise np.linalg.LinAlgError()

        self._M = np.linalg.inv(M0) * (1-a)
        self._weight_matrix_invalidated = False

        # end of def _prepare_exact_solution

    def _prepare_iterative_solution(self):
        # To get Dc and W_ot
        a = self._params.alpha
        [Ds, S_in, S_out, Wn_norm] = self._prepare_solution_common()

        # Coefficient matrix
        self._Dc = np.diag(1.0 / (a*(Ds+S_in+S_out) + (1-a)))

        # W(original + transposed) = Wn + Wn.T
        self._W_ot = Wn_norm + Wn_norm.T

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
            x_t2 = Dc.dot(a * W_ot.dot(x_t1) + (1 - a) * b)
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
# end of def class NormalizedGaussianSmoothing
