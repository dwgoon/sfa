
# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import pandas as pd

import sfa.base
from sfa.utils import FrozenClass

def create_algorithm(abbr):
    return PathwayWiring(abbr)
# end of def


class ParameterSet(FrozenClass):
    """
    Parameters of SignalPropagation algorithm.
    """

    def __init__(self):
        self.initialize()
        self._freeze()

    def initialize(self):
        self._max_path_length = 6
        self._weight = 0.5  # float value in (0, 1). The default value is 0.5.
        self._is_rel_change = False

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
    def is_rel_change(self):
        return self._is_rel_change

    @is_rel_change.setter
    def is_rel_change(self, val):
        if not isinstance(val, bool):
            raise TypeError("is_rel_change is bool type.")
        self._is_rel_change = val
# end of def class ParameterSet


class PathwayWiring(sfa.base.Algorithm):

    def __init__(self, abbr):
        super().__init__(abbr)
        self._name = "Feiglin's pathway wiring algorithm"

        self._params = ParameterSet()

        # The following members are assigned the instances in initialize()
        self._names_ba = None
        self._vals_ba = None
        self._iadj_to_idf = None
        self._result = sfa.base.Result()
    # end of def __init__

    def _initialize_network(self):
        """
        w = self._params.weight
        dg = self._data.dg
        A = self._data.A
        n2i = self._data.n2i

        # Assign weights for the edges
        for edge in dg.edges():
            src, tgt = edge
            sign = A[n2i[tgt], n2i[src]]
            dg.edge[src][tgt]['weight'] = sign*w
        """
        pass
    # end of def _initialize_network

    def _initialize_data(self):
        # N = self._data.A.shape[0]  # Number of state variables
        df_ba = self._data.df_ba  # Basal activity

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
        self._iadj_to_idf = [self._n2i[x] for x in
                             self._data.df_exp.columns]

    # end of _initialize_data

    def _apply_inputs(self, names, vals):
        if hasattr(self._data, 'inputs'):  # Input condition
            names.extend(self._data.inputs.keys())
            vals.extend(self._data.inputs.values())
        # end of if
    # end of def

    def compute(self):
        dg = self._data.dg
        n2i = self._data.n2i
        df_exp = self._data.df_exp  # Result of experiment

        # Simulation result
        sim_result = np.zeros(df_exp.shape, dtype=np.float)

        if self._params.is_rel_change:
            names_ba_se = []
            vals_ba_se = []
            self._apply_inputs(names_ba_se, vals_ba_se)
            x_cnt = self.wire(dg, names_ba_se, vals_ba_se, n2i)

        # Main loop of the simulation
        for i, names_ba_se in enumerate(self._names_ba):
            vals_ba_se = self._vals_ba[i]
            self._apply_inputs(names_ba_se, vals_ba_se)
            x_exp = self.wire(dg, names_ba_se, vals_ba_se, n2i)

            # Result of a single condition
            if self._params.is_rel_change:  # Use relative change
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

    def wire_single_path(self, dg, ba, path):
        F = ba
        w = self._params
        for i in range(len(path) - 1):
            src = path[i]
            tgt = path[i + 1]
            sign = dg.edge[src][tgt]['sign']
            F *= (sign*w)
        # end of for
        return F

    def wire_all_paths(self, dg, ba, src, tgt):
        paths = nx.all_simple_paths(dg, src, tgt)
        E = 0
        # Calculate the F for each path
        for i, path in enumerate(paths):
            F = self.wire_single_path(dg, ba, path)
            E += F
        # end of for

        # Apply the effect of perturbation on the target itself
        # if ptb == tgt:
        #    F = calc_F(dg, [tgt], w)
        #    E += F
        return E

    def wire(self, dg, names_ba_se, val_ba_se, n2i):
        """
        dg: NetworkX.DiGraph object including information of signs and weights
        names_ba_se: names of basal activities in a single experiment
        val_ba_se: values of basal activities in a single experiment
        n2i: name to index mapper
        """
        CE = np.zeros((dg.number_of_nodes(),), dtype=np.float)

        for tgt in dg.nodes_iter():
            Et = 0.0
            for i, src in enumerate(names_ba_se):
                ba = val_ba_se[i]
                # print(name_src, ba)
                Et += self.wire_all_paths(dg, ba, src, tgt)
                # end of for
            CE[n2i[tgt]] = Et
        # end of for
        return CE
    # end of def wire

# end of def class PathwayWiring
