
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
    return PathwayWiring(abbr)
# end of def




class PathwayWiring(sfa.base.Algorithm):

    class ParameterSet(FrozenClass):
        """
        Parameters of PathwayWiring algorithm.
        """

        def __init__(self):
            self.initialize()
            self._freeze()

        def initialize(self):
            self._max_path_length = None
            self._weight = 0.5  # float value in (0, 1). The default value is 0.5.
            self._use_rel_change = False
            self._no_inputs = True
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

        self._params = PathwayWiring.ParameterSet()

        # The following members are assigned the instances in initialize()
        self._names_ba = None
        self._vals_ba = None
        self._iadj_to_idf = None
        self._result = sfa.base.Result()
    # end of def __init__

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, mat):
        self._W = mat

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
    # end of def _initialize_network

    def _initialize_data(self):
        # N = self._data.A.shape[0]  # Number of state variables
        df_ba = self._data.df_ba  # Basal activity
        n2i = self.data.n2i

        self._names_ba = []
        self._vals_ba = []
        for i, row in enumerate(df_ba.iterrows()):
            row = row[1]
            list_name = []  # Names
            list_val = []  # Values
            for target in df_ba.columns[row.nonzero()]:
                list_name.append(target)
                list_val.append(row[target])
            # end of for
            self._names_ba.append(list_name)
            self._vals_ba.append(list_val)
        # end of for

        # For mapping from the indices of adj. matrix to those of DataFrame
        # (arrange the indices of adj. matrix according to df_exp.columns)
        self._iadj_to_idf = [n2i[x] for x in self._data.df_exp.columns]

        self._i2n = {idx: name for name, idx in n2i.items()}
    # end of _initialize_data

    def _apply_inputs(self, names, vals):
        if self._params.no_inputs:
            return

        # Input condition
        if hasattr(self._data, 'inputs') and self._data.inputs:
            names.extend(self._data.inputs.keys())
            vals.extend(self._data.inputs.values())
        # end of if
    # end of def

    def compute_batch(self):
        df_exp = self._data.df_exp  # Result of experiment

        # Simulation result
        sim_result = np.zeros(df_exp.shape, dtype=np.float)

        if self._params.use_rel_change:
            names_ba_se = []
            vals_ba_se = []
            self._apply_inputs(names_ba_se, vals_ba_se)
            x_cnt = self.wire(names_ba_se, vals_ba_se)

        # Main loop of the simulation
        for i, names_ba_se in enumerate(self._names_ba):
            vals_ba_se = self._vals_ba[i]  # 'se' means a 'single experiment'
            self._apply_inputs(names_ba_se, vals_ba_se)
            x_exp = self.wire(names_ba_se, vals_ba_se)

            # Result of a single condition
            if self._params.use_rel_change:  # Use relative change
                rel_change = ((x_exp - x_cnt) / np.abs(x_cnt))
                res_single = rel_change[self._iadj_to_idf]
            else:
                res_single = x_exp[self._iadj_to_idf]

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
        return self.wire(names_ba_se, val_ba_se)

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

    def wire_all_paths(self, dg, ba, src, tgt, getpath=False):
        mpl = self._params.max_path_length
        paths = nx.all_simple_paths(dg, src, tgt, mpl)
        E = 0

        if getpath:
            list_paths = []

        # Calculate the F for each path
        for i, path in enumerate(paths):
            F = self.wire_single_path(dg, ba, path)
            E += F
            if getpath:
                list_paths.append(path)
        # end of for

        # Apply the effect of perturbation on the target itself
        # if ptb == tgt:
        #    F = calc_F(dg, [tgt], w)
        #    E += F
        if getpath:
            return E, list_paths
        else:
            return E

    def wire(self, names_ba_se, val_ba_se, getpath=False):
        """
        names_ba_se: names of basal activities in a single experiment
        val_ba_se: values of basal activities in a single experiment
        """
        dg = self._dg
        n2i = self._data.n2i

        # The combined effects
        CE = np.zeros((dg.number_of_nodes(),), dtype=np.float)

        if getpath:
            list_paths = []

        for tgt in dg.nodes_iter():
            Et = 0.0
            for i, src in enumerate(names_ba_se):
                ba = val_ba_se[i]
                # print(name_src, ba)
                if getpath:
                    E, paths = self.wire_all_paths(dg, ba, src, tgt, getpath)
                    list_paths.extend(paths)
                    Et += E
                else:
                    Et += self.wire_all_paths(dg, ba, src, tgt, getpath)
                # end of for
            CE[n2i[tgt]] = Et
        # end of for

        if getpath:
            return CE, list_paths
        else:
            return CE
    # end of def wire

# end of def class PathwayWiring
