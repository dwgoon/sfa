# -*- coding: utf-8 -*-

import sys

if sys.version_info <= (2, 8):
    from builtins import super


import numpy as np
import pandas as pd

import sfa.base
import sfa.utils


class NetworkPropagationParameterSet(sfa.base.ParameterSet):
    """
    An object that deals with the parameters
    of ``sfa.algorithms.np.NetworkPropagation`` base algorithm.

    Attributes
    ----------
    alpha : float
    lim_iter : int
    apply_weight_norm : bool
    use_rel_change : bool
    exsol_forbidden : bool
    no_inputs : bool
    """

    def __init__(self):
        self.initialize()
        self._freeze()

    def initialize(self):
        """Initialize the parameters with default values.
        """
        self._alpha = 0.5  # float value in (0, 1). The default value is 0.5.
        self._lim_iter = 1000
        self._apply_weight_norm = False
        self._use_rel_change = False
        self._exsol_forbidden = False
        self._no_inputs = False

    @property
    def alpha(self):
        r"""Hyperparameter, :math:`\alpha` ~ (0, 1).
         It controls the portion of signal flow
         in determining the activity.
         The default value is 0.5.
        """
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
    def lim_iter(self):
        """Number of maximum iterations in the iterative method.
        """
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
    def apply_weight_norm(self):
        """Apply the link weight normalization.
        """
        return self._apply_weight_norm

    @apply_weight_norm.setter
    def apply_weight_norm(self, val):
        if not isinstance(val, bool):
            raise TypeError(
                "apply_weight_norm should be a bool type value.")
        self._apply_weight_norm = val

    @property
    def use_rel_change(self):
        """Use relative change for prediction.
        """

        return self._use_rel_change

    @use_rel_change.setter
    def use_rel_change(self, val):
        if not isinstance(val, bool):
            raise TypeError("use_rel_change should be a bool type value.")
        self._use_rel_change = val

    @property
    def exsol_forbidden(self):
        """Forbid the propagation computation
           based on the exact solution.
           In other words, use the iterative method.
        """
        return self._exsol_forbidden

    @exsol_forbidden.setter
    def exsol_forbidden(self, val):
        if not isinstance(val, bool):
            raise TypeError("exsol_forbidden should be boolean type.")

        self._exsol_forbidden = val

    @property
    def no_inputs(self):
        """Do not apply the effects of inputs in a given network.
        """
        return self._no_inputs

    @no_inputs.setter
    def no_inputs(self, val):
        if not isinstance(val, bool):
            raise TypeError("no_inputs is bool type.")
        self._no_inputs = val
# end of def class ParameterSet


class NetworkPropagation(sfa.base.Algorithm):
    """A base class that defines the basic functionality of
       network propagation algorithms.

    Attributes
    ----------


    """

    def __init__(self, abbr):
        """Constructor of NetworkPropagation.
        """
        super().__init__(abbr)
        self._name = "Abstract base class for network propagation algorithms"
        self._params = NetworkPropagationParameterSet()

        # The following members are assigned the instances in initialize()
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

    # end of _W.setter

    def initialize_network(self):

        # Matrix normalization for getting transition matrix
        if self._params.apply_weight_norm:
            self.W = sfa.utils.normalize(self.data.A)
        else:
            self.W = np.array(self.data.A, dtype=np.float)

        self._check_dimension(self.W, "transition matrix")

        if not self.params.exsol_forbidden:
            # Try to prepare the exact solution
            try:
                self.prepare_exact_solution()
                self._check_dimension(self._M, "exact solution matrix")
                self._exsol_avail = True
            except np.linalg.LinAlgError:
                pass

        if not self._exsol_avail:
            self.prepare_iterative_solution()
            self._exsol_avail = False

    # end of def _initialize_network

    def _check_dimension(self, mat, mat_name):
        """Check whether a given matrix is a square matrix.
        """
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("The %s should be square matrix." % (mat_name))
    # end of def _check_dimension

    def initialize_basal_activity(self):
        N = self.data.A.shape[0]  # Number of state variables
        self._b = np.zeros(N)
    # end of def

    def apply_inputs(self, inds, vals):
        """

        Parameters
        ----------

        """
        if self._params.no_inputs:
            return

        # Input condition
        if hasattr(self.data, 'inputs') and self.data.inputs:
            inds_inputs = [self.data.n2i[inp] for inp in self.data.inputs]
            vals_inputs = [val for val in self.data.inputs.values()]
            inds.extend(inds_inputs)
            vals.extend(vals_inputs)
            # end of if

    # end of def apply_inputs

    def apply_perturbations(self, targets, inds, vals, W_ptb=None):
        if self.data.has_link_perturb and W_ptb is None:
            raise ValueError("Weight matrix for perturbation is necessary for "
                             "the data including link type perturbations.")

        for target in targets:
            if self.data.df_ptb is not None:
                type_ptb = self.data.df_ptb.loc[target, "Type"]
                val_ptb = self.data.df_ptb.loc[target, "Value"]
            else:
                type_ptb = 'node'
                val_ptb = -1

            if type_ptb == 'node':
                inds.append(self.data.n2i[target])
                vals.append(val_ptb)
            elif type_ptb == 'link':
                idx = self.data.n2i[target]
                W_ptb[:, idx] *= val_ptb
            elif type_ptb == 'isolation':
                idx = self.data.n2i[target]
                W_ptb[:, idx] *= val_ptb
                W_ptb[idx, :] *= val_ptb
            else:
                raise ValueError("Undefined perturbation type: %s" % (type_ptb))

    # end of def apply_perturbations

    def compute_batch(self):

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
            x_exp = self.compute(b)

            # Result of a single condition
            if self._params.use_rel_change:  # Use relative change
                x_diff = (x_exp - x_cnt)
                rel_change = x_diff
                res_single = rel_change[self.data.iadj_to_idf]
            else:
                res_single = x_exp[self.data.iadj_to_idf]

            sim_result[i, :] = res_single
            b[inds_ba] = b_store
        # end of for

        self.W = W_cnt

        df_sim = pd.DataFrame(sim_result,
                              index=df_exp.index,
                              columns=df_exp.columns)

        # Get the result of elements in the columns of df_exp.
        self._result.df_sim = df_sim[df_exp.columns]

    # end of def compute_batch

    def prepare_exact_solution(self):
        """Prepare to get the matrix for the exact solution.
        """
    # end of def _prepare_exact_solution

    def prepare_iterative_solution(self):
        """Prepare to get the solution from the iterative method.
        """
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
        """Obtain the activity at steady-state
        based on the exact solution of network propagation.

        Parameters
        ----------
        b : numpy.ndarray
            1D array of basal activity.

        Returns
        -------
        x : numpy.ndarray
            The exact solution of the activity at steady-state
            in 1D array.
        """
        raise NotImplementedError("propagate_exact is not implemented")

    def propagate_iterative(self,
                            W,
                            xi,
                            b,
                            a=0.5,
                            lim_iter=1000,
                            tol=1e-5,
                            get_trj=False):

        r"""Compute network propagation based on the iterative method.
        This method should be used if we want to obtain the trajectory.

        Parameters
        ----------
        W: numpy.ndarray
            2D array for weight matrix.
        xi: numpy.ndarray
            1D array for initial state.
        b: numpy.ndarray
            1D array for basal activity.
        a: real number, optional
            Hyperparameter, :math:`\alpha`, ~ (0, 1).
            The default value is 0.5.
        lim_iter: int, optional
            Number of maximum iterations in the iterative method.
            Computation terminates when the iteration reaches ``lim_iter``.
            The default value is 1000.
        tol: float, optional
            Tolerance for terminating iteration.
            Iteration continues, if Frobenius norm of
            :math:`x(t+1)-x(t)` is greater than ``tol``.
            The default value is 1e-5.
        get_trj: bool, optional
            Determine whether the trajectory of the state is returned.
            If get_trj is true, the trajectory is returned.

        Returns
        -------
        x : numpy.ndarray
            1D array of the activity after the computation.
        trj : numpy.ndarray
            2D array where the row represents a state of the activity.

        See also
        --------
        """
        raise NotImplementedError("propagate_iterative is not implemented")
    # end of def propagate_iterative

# end of def class NetworkPropagation
