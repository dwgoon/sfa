
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
        self._weight = 0.5  # float value in (0, 1). The default value is 0.5.
        self._is_rel_change = False

    @property
    def weight(self):
        return self._alpha

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
        self._b = None
        self._ind_ba = None
        self._val_ba = None
        self._iadj_to_idf = None
        self._P = None

        self._result = sfa.base.Result()

    # end of def __init__

    def propagate(self, b):

        """
        if self._exsol_avail:
            return self.propagate_exact(b)
        else:
            alpha = self._params.alpha
            P = self._P
            b = self._b
            x_ss, _ = self.propagate_iterative(P, b, b, a=alpha)
        """

        return x_ss  # x at steady-state (i.e., staionary state)
    # end of def propagate

    def calc_F(self, dg, path, w):
        F = -1
        len_path = len(path)
        for i in range(len_path - 1):
            src = path[i]
            tgt = path[i + 1]
            sign = dg.edge[src][tgt]['sign']
            F *= (sign * w)
        # end of for
        return F

    def single_ptb_calc_E(self, dg, ptb, tgt, w):
        paths = nx.all_simple_paths(dg, ptb, tgt)
        E = 0
        # Calculate the F for each path
        for i, p in enumerate(paths):
            F = self.calc_F(dg, p, w)
            E += F
        # end of for

        # Apply the effect of perturbation on the target itself
        # if ptb == tgt:
        #    F = calc_F(dg, [tgt], w)
        #    E += F

        return E

    def calc_CE(self, dg, ptbs, w=0.5, node_names=None):
        # CE means the 'combined effect' of the paper
        arr_CE = np.zeros((dg.number_of_nodes(),), dtype=np.float)

        if node_names is None:
            node_names = dg.nodes()

        for i, output in enumerate(node_names):
            Et = 0
            for ptb in ptbs:
                Et += self.single_ptb_calc_E(dg, ptb, output, w)
            # end of for
            arr_CE[i] = Et

        # end of for
        return arr_CE
# end of def class PathwayWiring