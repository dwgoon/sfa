# -*- coding: utf-8 -*-

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

    def _normalize(self, A):
        return A  # No norm.

    def _prepare_exact_solution(self):
        a = self._params.alpha
        W = self._P
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

        self._M = np.linalg.inv(M0) * (1-a)

    def propagate_exact(self, b):
        return self._M.dot(b)

    def propagate_iterative(self,
                            P,
                            xi,
                            b,
                            a=0.5,
                            lim_iter=1000,
                            tol=1e-5,
                            notrj=True):
        n = P.shape[0]
        # Initial values
        x0 = np.zeros((n,), dtype=np.float)
        x0[:] = xi

        x_t1 = x0.copy()
        trj_x = []

        # Record the initial states
        trj_x.append(x_t1.copy())

        # To get Dc and W_ot
        W = P
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
            if not notrj:
                trj_x.append(x_t2)

            # Update the state
            x_t1 = x_t2.copy()
        # end of for

        if notrj is True:
            return x_t2, num_iter
        else:
            return x_t2, np.array(trj_x)


    # end of def propagate_iterative

# end of def class GaussianSmoothing