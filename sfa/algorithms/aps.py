
# -*- coding: utf-8 -*-
import sys
if sys.version_info <= (2, 8):
    from builtins import super

import networkx as nx
import numpy as np
import pandas as pd

import sfa.base
from sfa.utils import FrozenClass

def create_algorithm(abbr):
    return AcyclicPathSummation(abbr)
# end of def

np.seterr(all='raise')


class AcyclicPathSummation(sfa.base.Algorithm):

    class ParameterSet(FrozenClass):
        """
        Parameters of AcyclicPathSummation algorithm.
        """

        def __init__(self):
            self.initialize()
            self._freeze()

        def initialize(self):
            self._max_path_length = None
            self._weight = 0.5  # float value in (0, 1). The default value is 0.5.
            self._use_rel_change = False
            self._no_inputs = False
            self._use_local_weight = False
            self._apply_weight_norm = False

        @property
        def max_path_length(self):
            return self._max_path_length

        @max_path_length.setter
        def max_path_length(self, val):
            self._max_path_length = val

        @property
        def weight(self):
            return self._weight

        @weight.setter
        def weight(self, val):
            if not isinstance(val, float):
                raise TypeError("weight is a float type value in (0,1).")
            elif (val <= 0.0) or (val >= 1.0):
                raise ValueError("weight should be within (0,1).")
            else:
                self._weight = val


        # alpha is an alias of weight.
        @property
        def alpha(self):
            return self.weight

        @alpha.setter
        def alpha(self, val):
            self.weight = val

        @property
        def use_rel_change(self):
            return self._use_rel_change

        @use_rel_change.setter
        def use_rel_change(self, val):
            if not isinstance(val, bool):
                raise TypeError("use_rel_change is bool type.")
            self._use_rel_change = val

        @property
        def no_inputs(self):
            return self._no_inputs

        @no_inputs.setter
        def no_inputs(self, val):
            if not isinstance(val, bool):
                raise TypeError("no_inputs is bool type.")
            self._no_inputs = val

        @property
        def use_local_weight(self):
            return self._use_local_weight

        @use_local_weight.setter
        def use_local_weight(self, val):
            if not isinstance(val, bool):
                raise TypeError("use_local_weight is bool type.")
            self._use_local_weight = val

        @property
        def apply_weight_norm(self):
            return self._apply_weight_norm

        @apply_weight_norm.setter
        def apply_weight_norm(self, val):
            if not isinstance(val, bool):
                raise TypeError(
                    "apply_weight_norm should be a bool type value.")
            self._apply_weight_norm = val

    # end of def class ParameterSet

    def __init__(self, abbr):
        super().__init__(abbr)
        self._name = "Feiglin's pathway wiring algorithm"

        self._params = AcyclicPathSummation.ParameterSet()

        # The following members are assigned the instances in initialize()
        #self._names_ba = None
        #self._vals_ba = None
        #self._iadj_to_idf = None

        self._weight_matrix_invalidated = True

        self._result = sfa.base.Result()
    # end of def __init__

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, mat):
        self._W = mat
        self._weight_matrix_invalidated = True


    def _initialize_network(self):
        self._dg = self._data.dg  # Initialization of dg
        A = self._data.A
        n2i = self._data.n2i

        if self._params.apply_weight_norm:
            self._W = sfa.utils.normalize(A)

        if self._params.use_local_weight:
            for edge in self._dg.edges():
                src, tgt = edge
                isrc = n2i[src]
                itgt = n2i[tgt]
                self._dg.edge[src][tgt]['weight'] = self._W[itgt, isrc]
        else:  # Use global weight
            # Assign weights for the edges
            for edge in self._dg.edges():
                src, tgt = edge
                self._dg.edge[src][tgt]['weight'] = self._params.weight
        # end of if-else

        self._weight_matrix_invalidated = True
    # end of def _initialize_network

    # def _initialize_data(self):
    #     # N = self._data.A.shape[0]  # Number of state variables
    #     # df_conds = self._data.df_conds  # Basal activity
    #     # n2i = self.data.n2i
    #     #
    #     # self._names_ba = []
    #     # self._vals_ba = []
    #     # for i, row in enumerate(df_conds.iterrows()):
    #     #     row = row[1]
    #     #     list_name = []  # Names
    #     #     list_val = []  # Values
    #     #     for target in df_conds.columns[row.nonzero()]:
    #     #         list_name.append(target)
    #     #         list_val.append(row[target])
    #     #     # end of for
    #     #     self._names_ba.append(list_name)
    #     #     self._vals_ba.append(list_val)
    #     # # end of for
    #     #
    #     # # For mapping from the indices of adj. matrix to those of DataFrame
    #     # # (arrange the indices of adj. matrix according to df_exp.columns)
    #     # self._iadj_to_idf = [n2i[x] for x in self._data.df_exp.columns]
    #     #
    #     # self._i2n = {idx: name for name, idx in n2i.items()}
    #     pass
    # # end of _initialize_data

    def _apply_inputs(self, names, vals):
        if self._params.no_inputs:
            return

        # Input condition
        if hasattr(self._data, 'inputs') and self._data.inputs:
            names.extend(self._data.inputs.keys())
            vals.extend(self._data.inputs.values())
        # end of if
    # end of def

    def _apply_perturbations(self, targets, names, vals, dg):
        for target in targets:
            type_ptb = self.data.df_ptb.ix[target, "Type"]
            val_ptb = self.data.df_ptb.ix[target, "Value"]
            if type_ptb == 'node':
                names.append(target)
                vals.append(val_ptb)
            elif type_ptb == 'link':
                for downstream, attr in dg.edge[target].items():
                    attr["weight"] *= val_ptb
            else:
                raise ValueError("Undefined perturbation type: %s"%(type_ptb))

    def compute_batch(self):
        df_exp = self._data.df_exp  # Result of experiment

        # Simulation result
        sim_result = np.zeros(df_exp.shape, dtype=np.float)

        if self._params.use_rel_change:
            names_ba_se = []
            vals_ba_se = []
            self._apply_inputs(names_ba_se, vals_ba_se)
            x_cnt = self.wire(self._dg, names_ba_se, vals_ba_se)
        # end of if

        # Main loop of the simulation
        for i, targets_ptb in enumerate(self._data.names_ptb):
            dg_ptb = self._dg.copy()
            names_ba_se = []
            vals_ba_se = []  # 'se' means a 'single experiment'
            self._apply_inputs(names_ba_se, vals_ba_se)
            self._apply_perturbations(targets_ptb,
                                      names_ba_se, vals_ba_se, dg_ptb)

            x_exp = self.wire(dg_ptb, names_ba_se, vals_ba_se)

            # Result of a single condition
            if self._params.use_rel_change:  # Use relative change
                rel_change = x_exp - x_cnt
                res_single = rel_change[self._data.iadj_to_idf]
            else:
                res_single = x_exp[self._data.iadj_to_idf]

            sim_result[i, :] = res_single
        # end of for

        df_sim = pd.DataFrame(sim_result,
                              index=df_exp.index,
                              columns=df_exp.columns)

        # Get the result of elements in the columns of df_exp.
        self._result.df_sim = df_sim[df_exp.columns]
    # end of def compute

    def compute(self, b):
        i2n = self._i2n
        names_ba_se = []
        val_ba_se = []
        for i, val in enumerate(b):
            names_ba_se.append(i2n[i])
            val_ba_se.append(val)
        # end of for
        return self.wire(self._dg, names_ba_se, val_ba_se)

    def wire_single_path(self, dg, ba, path):
        F = ba
        for i in range(len(path) - 1):
            src = path[i]
            tgt = path[i + 1]
            sign = dg.edge[src][tgt]['sign']
            w = dg.edge[src][tgt]['weight']
            F *= (sign*w)

        # end of for
        return F

    def wire_all_paths(self, dg, ba, src, tgt, get_path=False):
        E = 0

        if get_path:
            list_paths = []

        # Apply the effect of perturbation on the target itself
        if src == tgt:
           # pass
           F = self.wire_single_path(dg, ba, [tgt])
           E += F

           if get_path:
               list_paths.append([src])

        else:
            mpl = self._params.max_path_length
            paths = nx.all_simple_paths(dg, src, tgt, mpl)

            # Calculate the F for each path
            for i, path in enumerate(paths):
                F = self.wire_single_path(dg, ba, path)
                E += F
                if get_path:
                    list_paths.append(path)
            # end of for

        if get_path:
            return E, list_paths
        else:
            return E

    def wire(self, dg, names_ba_se, val_ba_se, get_path=False):
        """
        names_ba_se: names of basal activities in a single experiment
        val_ba_se: values of basal activities in a single experiment
        """
        #dg = self._dg
        n2i = self._data.n2i

        # The combined effects
        CE = np.zeros((dg.number_of_nodes(),), dtype=np.float)

        if get_path:
            list_paths = []

        for tgt in dg.nodes_iter():
            Et = 0.0
            for i, src in enumerate(names_ba_se):

                ba = val_ba_se[i]
                if get_path:
                    E, paths = self.wire_all_paths(dg, ba, src, tgt, get_path)
                    list_paths.extend(paths)
                    Et += E
                else:
                    Et += self.wire_all_paths(dg, ba, src, tgt, get_path)
                # end of for
            CE[n2i[tgt]] = Et
        # end of for

        if get_path:
            return CE, list_paths
        else:
            return CE
    # end of def wire

# end of def class AcyclicPathSummation
