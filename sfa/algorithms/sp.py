# -*- coding: utf-8 -*-
"""
@author: dwlee
"""

import numpy as np
import pandas as pd

import sfa.base
from sfa.utils import FrozenClass


def create_algorithm(abbr):
    return SignalPropagation(abbr)
# end of def

class ParameterSet(FrozenClass):
    """
    Parameters of SignalPropagation algorithm.
    """

    def __init__(self):
        self.initialize()
        self._freeze()

    def initialize(self):
        self._alpha = 0.5  # float value in (0, 1). The default value is 0.5.
        self._is_rel_change = False

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, val):
        if not isinstance(val, float):
            raise TypeError("alpha is a float type value in (0,1).")
        elif (val <= 0.0) or (val >= 1.0):
            raise ValueError("alpha should be within (0,1).")
        else:
            self._alpha = val

    @property
    def is_rel_change(self):
        return self._is_rel_change

    @is_rel_change.setter
    def is_rel_change(self, val):
        if not isinstance(val, bool):
            raise TypeError("is_rel_change is bool type.")
        self._is_rel_change = val
# end of def class ParameterSet


class SignalPropagation(sfa.base.Algorithm):
    def __init__(self, abbr):
        super().__init__(abbr)
        self._name = "Signal propagation algorithm"
        self._params = ParameterSet()

        # The following members are assigned the instances in initialize()
        self._b = None
        self._ind_ba = None
        self._val_ba = None
        self._iadj_to_idf = None
        self._W = None

        self._exsol_avail = False  # The exact solution is available.
        self._M = None  # A matrix for getting the exact solution.

        self._result = sfa.base.Result()
    # end of def __init__

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, mat):
        self._W = mat

    def _initialize_network(self):
        # Matrix normalization for getting transition matrix
        self._W = self.normalize(self._data.A)
        self._check_dimension(self._W, "transition matrix")
        # Try to prepare the exact solution
        try:
            self._M = self._prepare_exact_solution()
            self._check_dimension(self._M, "exact solution matrix")
            self._exsol_avail = True
        except np.linalg.LinAlgError:
            self._exsol_avail = False
    # end of def _initialize_network

    def _check_dimension(self, mat, mat_name):
        # Check whether a given matrix is a square matrix.
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("The %s should be square matrix."%(mat_name))
    # end of def _check_dimension

    def _initialize_data(self):
        N = self._data.A.shape[0]  # Number of state variables
        n2i = self._data.n2i  # Name to index mapper
        df_ba = self._data.df_ba  # Basal activity

        #self._b = np.zeros(N, dtype=np.float)
        self._b = np.finfo(np.float).eps * np.ones(N)
        self._inds_ba = []  # Indices (inds)
        self._vals_ba = []  # Values (vals)
        for i, row in enumerate(df_ba.iterrows()):
            row = row[1]
            list_ind = []  # Indices
            list_val = []  # Values
            for target in df_ba.columns[row.nonzero()]:
                list_ind.append(n2i[target])
                list_val.append(row[target])
            # end of for

            self._inds_ba.append(list_ind)
            self._vals_ba.append(list_val)
        # end of for

        # For mapping from the indices of adj. matrix to those of DataFrame
        # (arrange the indices of adj. matrix according to df_exp.columns)
        self._iadj_to_idf = [n2i[x] for x in self._data.df_exp.columns]
    # end of _initialize_data

    def _apply_inputs(self, b):
        # Input condition
        if hasattr(self._data, 'inputs') and self._data.inputs:
            ind_inputs = [self._data.n2i[inp] for inp in self._data.inputs]
            val_inputs = [val for val in self._data.inputs.values()]
            b[ind_inputs] = val_inputs
        # end of if
    # end of def _apply_inputs

    def compute_batch(self):
        """Algorithm perform the computation with the given data"""
        df_exp = self._data.df_exp  # Result of experiment

        # Simulation result
        sim_result = np.zeros(df_exp.shape, dtype=np.float)

        b = self._b

        # if hasattr(self._data, 'inputs'):  # Input condition
        #     ind_inputs = [self._data.n2i[inp] for inp in self._data.inputs]
        #     val_inputs = [val for val in self._data.inputs.values()]
        #     b[ind_inputs] = val_inputs
        # # end of if

        if self._params.is_rel_change:
            self._apply_inputs(b)
            x_cnt = self.compute(b)

        # Main loop of the simulation
        for i, ind_ba in enumerate(self._inds_ba):
            ind_ba = self._inds_ba[i]
            b_store = b[ind_ba][:]
            self._apply_inputs(b)  # Apply the input condition
            b[ind_ba] = self._vals_ba[i]  # Basal activity

            x_exp = self.compute(b)

            # Result of a single condition
            if self._params.is_rel_change:  # Use relative change
                rel_change = ((x_exp - x_cnt) / np.abs(x_cnt))
                res_single = rel_change[self._iadj_to_idf]
            else:
                res_single = x_exp[self._iadj_to_idf]

            sim_result[i, :] = res_single
            b[ind_ba] = b_store
        # end of for

        df_sim = pd.DataFrame(sim_result,
                              index=df_exp.index,
                              columns=df_exp.columns)

        # Get the result of elements in the columns of df_exp.
        self._result.df_sim = df_sim[df_exp.columns]
    # end of def compute

    def normalize(self, A, norm_in=True, norm_out=True):

        # Check whether A is a square matrix
        if A.shape[0] != A.shape[1]:
            raise ValueError(
                "The A (adjacency matrix) should be square matrix.")

        # Build propagation matrix (aka. transition matrix) W from A
        W = A.copy()

        # Norm. in-degree
        if norm_in == True:
            sum_col_A = np.abs(A).sum(axis=0)
            sum_col_A[sum_col_A == 0] = 1
            if norm_out == False:
                Dc = 1 / sum_col_A
            else:
                Dc = 1 / np.sqrt(sum_col_A)
            # end of else
            W = Dc * W  # This is not matrix multiplication

        # Norm. out-degree
        if norm_out == True:
            sum_row_A = np.abs(A).sum(axis=1)
            sum_row_A[sum_row_A == 0] = 1
            if norm_in == False:
                Dr = 1 / sum_row_A
            else:
                Dr = 1 / np.sqrt(sum_row_A)
            # end of row
            W = np.multiply(W, np.mat(Dr).T)
            # Converting np.mat to ndarray
            # does not cost a lot.
            W = W.A
        # end of if
        """
        The normalization above is the same as the follows:
        >>> np.diag(Dr).dot(A.dot(np.diag(Dc)))
        """
        return W

    # end of def normalize

    def _prepare_exact_solution(self):
        """
        Prepare to get the matrix for the exact solution:

        x(t+1) = a*W.dot(x(t)) + (1-a)*b, where a is alpha.

        When t -> inf, both x(t+1) and x(t) converges to the stationary state.

        Then, s = aP*s + (1-a)b
              (I-aP)*s = (1-a)b
              s = (I-aP)^-1 * (1-a)b
              s = M*b, where M is (1-a)(I-aP)^-1.

        This method is to get the matrix, M for preparing the exact solution
        """
        W = self._W
        a = self._params.alpha
        M0 = np.eye(W.shape[0]) - a*W
        return (1-a)*np.linalg.inv(M0)
    # end of def _prepare_exact_solution

    def compute(self, b):
        if self._exsol_avail:
            return self.propagate_exact(b)
        else:
            alpha = self._params.alpha
            W = self._W
            x_ss, _ = self.propagate_iterative(W, b, b, a=alpha)
            return x_ss  # x at steady-state (i.e., staionary state)
    # end of def compute

    def propagate_exact(self, b):
        return self._M.dot(b)

    def propagate_iterative(self,
                            W,
                            xi,
                            b,
                            a=0.5,
                            lim_iter=1000,
                            tol=1e-5,
                            gettrj=False):
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
            if gettrj:
                trj_x.append(x_t2)

            # Update the state
            x_t1 = x_t2.copy()
        # end of for

        if gettrj is False:
            return x_t2, num_iter
        else:
            return x_t2, np.array(trj_x)

    # end of def compute

# end of def class SignalPropagation
