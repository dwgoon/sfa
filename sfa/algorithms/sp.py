# -*- coding: utf-8 -*-

import sys
if sys.version_info <= (2, 8):
    from builtins import super

import numpy as np
import pandas as pd


import sfa.base
import sfa.utils


def create_algorithm(abbr):
    return SignalPropagation(abbr)
# end of def


class SignalPropagation(sfa.base.Algorithm):

    class ParameterSet(sfa.utils.FrozenClass):
        """
        Parameters of SignalPropagation algorithm.
        """
        def __init__(self):
            self.initialize()
            self._freeze()

        def initialize(self):
            self._alpha = 0.5  # float value in (0, 1). The default value is 0.5.
            self._apply_weight_norm = True
            self._use_rel_change = False
            self._exsol_forbidden = False
            self._lim_iter = 1000
            self._no_inputs = False


        @property
        def alpha(self):
            return self._alpha

        @alpha.setter
        def alpha(self, val):
            if not isinstance(val, float):
                raise TypeError("alpha should be a float type value in (0,1).")
            elif (val <= 0.0) or (val >= 1.0):
                raise ValueError("alpha should be within (0,1).")
            else:
                self._alpha = val

        @property
        def apply_weight_norm(self):
            return self._apply_weight_norm

        @apply_weight_norm.setter
        def apply_weight_norm(self, val):
            if not isinstance(val, bool):
                raise TypeError(
                    "apply_weight_norm should be a bool type value.")
            self._apply_weight_norm = val

        @property
        def use_rel_change(self):
            return self._use_rel_change

        @use_rel_change.setter
        def use_rel_change(self, val):
            if not isinstance(val, bool):
                raise TypeError("use_rel_change should be a bool type value.")
            self._use_rel_change = val

        @property
        def exsol_forbidden(self):
            return self._exsol_forbidden

        @exsol_forbidden.setter
        def exsol_forbidden(self, val):
            if not isinstance(val, bool):
                raise TypeError("exsol_forbidden should be boolean type.")

            self._exsol_forbidden = val

        @property
        def lim_iter(self):
            return self._lim_iter

        @lim_iter.setter
        def lim_iter(self, val):
            if not isinstance(val, int):
                raise TypeError("lim_iter is a integer type value.")
            elif val < 0:
                raise ValueError("lim_iter should be greater than 0.")
            else:
                self._lim_iter = val

        @property
        def no_inputs(self):
            return self._no_inputs

        @no_inputs.setter
        def no_inputs(self, val):
            if not isinstance(val, bool):
                raise TypeError("no_inputs is bool type.")
            self._no_inputs = val
    # end of def class ParameterSet



    def __init__(self, abbr):
        super().__init__(abbr)
        self._name = "Signal propagation algorithm"
        self._params = self.ParameterSet()

        # The following members are assigned the instances in initialize()
        #self._ind_ba = None
        #self._val_ba = None
        #self._iadj_to_idf = None
        self._W = None
        self._b = None

        self._exsol_avail = False  # The exact solution is available.
        self._M = None  # A matrix for getting the exact solution.
        self._weight_matrix_invalidated = True

        self._result = sfa.base.Result()
    # end of def __init__

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, vec):
        self._b = vec

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, mat):
        self._W = mat
        self._weight_matrix_invalidated = True
    # end of W.setter

    def _initialize_network(self):
        # Matrix normalization for getting transition matrix
        if self._params.apply_weight_norm:
            self.W = sfa.utils.normalize(self.data.A)
        else:
            self.W = np.array(self.data.A, dtype=np.float)

        self._check_dimension(self.W, "transition matrix")

        if not self.params.exsol_forbidden:
            # Try to prepare the exact solution
            try:
                self._prepare_exact_solution()
                self._check_dimension(self._M, "exact solution matrix")
                self._exsol_avail = True
            except np.linalg.LinAlgError:
                pass

        if not self._exsol_avail:
            self._prepare_iterative_solution()
            self._exsol_avail = False
    # end of def _initialize_network

    def _check_dimension(self, mat, mat_name):
        # Check whether a given matrix is a square matrix.
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("The %s should be square matrix."%(mat_name))
    # end of def _check_dimension
            
    def _initialize_basal_activity(self):
        N = self.data.A.shape[0]  # Number of state variables
        self._b = np.zeros(N)
        #self._b = np.finfo(np.float).eps * np.ones(N)
    # end of def
        
    # def _initialize_data(self):
    #     # n2i = self._data.n2i  # Name to index mapper
    #     # df_conds = self._data.df_conds  # Basal activity
    #     #
    #     # self._inds_ba = []  # Indices (inds)
    #     # self._vals_ba = []  # Values (vals)
    #     # for i, row in enumerate(df_conds.iterrows()):
    #     #     row = row[1]
    #     #     list_ind = []  # Indices
    #     #     list_val = []  # Values
    #     #     for target in df_conds.columns[row.nonzero()]:
    #     #         list_ind.append(n2i[target])
    #     #         list_val.append(row[target])
    #     #     # end of for
    #     #
    #     #     self._inds_ba.append(list_ind)
    #     #     self._vals_ba.append(list_val)
    #     # # end of for
    #     #
    #     # # For mapping from the indices of adj. matrix to those of DataFrame
    #     # # (arrange the indices of adj. matrix according to df_exp.columns)
    #     # self._iadj_to_idf = [n2i[x] for x in self._data.df_exp.columns]
    #     pass
    # # end of _initialize_data

    def apply_inputs(self, inds, vals):
        if self._params.no_inputs:
            return

        # Input condition
        if hasattr(self.data, 'inputs') and self.data.inputs:
            inds_inputs = [self.data.n2i[inp] for inp in self.data.inputs]
            vals_inputs = [val for val in self.data.inputs.values()]
            # b[ind_inputs] = val_inputs
            inds.extend(inds_inputs)
            vals.extend(vals_inputs)
        # end of if
    # end of def apply_inputs

    def apply_perturbations(self, targets, inds, vals, W_ptb=None):
        if self.data.has_link_perturb and W_ptb is None:
            raise ValueError("Weight matrix for perturbation is necessary for "
                             "the data including link type perturbations.")

        for target in targets:
            type_ptb = self.data.df_ptb.ix[target, "Type"]
            val_ptb = self.data.df_ptb.ix[target, "Value"]
            if type_ptb == 'node':
                inds.append(self.data.n2i[target])
                vals.append(val_ptb)
            elif type_ptb == 'link':
                idx = self.data.n2i[target]
                W_ptb[:, idx] *= val_ptb
                # try:
                #     W_ptb[:, idx] *= val_ptb
                # except TypeError:
                #     print (W_ptb.dtype)
                #     print (val_ptb)

            else:
                raise ValueError("Undefined perturbation type: %s"%(type_ptb))
    # end of def apply_perturbations

    def compute_batch(self):
        """Algorithm perform the computation with the given data"""
        df_exp = self.data.df_exp  # Result of experiment

        # Simulation result
        sim_result = np.zeros(df_exp.shape, dtype=np.float)

        b = self._b

        if self._params.use_rel_change:
            inds_ba = []  # Indices of nodes to be perturbed
            vals_ba = []  # Basal activity
            self.apply_inputs(inds_ba, vals_ba)
            b[inds_ba] = vals_ba
            x_cnt = self.compute(b)

        W_cnt = self.W

        # Main loop of the simulation
        for i, targets_ptb in enumerate(self.data.names_ptb):
            inds_ba = []  # Indices of nodes to be perturbed
            vals_ba = []  # Basal activity
            self.apply_inputs(inds_ba, vals_ba)  # Apply the input condition

            if self.data.has_link_perturb:
                W_ptb = W_cnt.copy()
                self.apply_perturbations(targets_ptb, inds_ba, vals_ba, W_ptb)
                self.W = W_ptb
            else:
                self.apply_perturbations(targets_ptb, inds_ba, vals_ba)

            b_store = b[inds_ba]
            b[inds_ba] = vals_ba

            # if self.data.has_link_perturb:
            #     alpha = self._params.alpha
            #     lim_iter = self._params.lim_iter
            #     x_exp, _ = self.propagate_iterative(W_ptb, b, b,
            #                                         a=alpha,
            #                                         lim_iter=lim_iter)
            # else:  # Data has a link perturbation
            x_exp = self.compute(b)

            # Result of a single condition
            if self._params.use_rel_change:  # Use relative change
                x_diff = (x_exp - x_cnt)
                rel_change = x_diff
                res_single = rel_change[self.data.iadj_to_idf]
            else:
                res_single = x_exp[self._iadj_to_idf]

            sim_result[i, :] = res_single
            b[inds_ba] = b_store
        # end of for

        self.W = W_cnt

        df_sim = pd.DataFrame(sim_result,
                              index=df_exp.index,
                              columns=df_exp.columns)

        # Get the result of elements in the columns of df_exp.
        self._result.df_sim = df_sim[df_exp.columns]
    # end of def compute

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

    def compute(self, b):
        if self.params.exsol_forbidden is True \
           or self._exsol_avail is False:
            alpha = self._params.alpha
            W = self.W
            lim_iter = self._params.lim_iter
            x_ss, _ = self.propagate_iterative(W, b, b, a=alpha,
                                               lim_iter=lim_iter)
            return x_ss  # x at steady-state (i.e., stationary state)
        else:
            return self.propagate_exact(b)

    # end of def compute

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
