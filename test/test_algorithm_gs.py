# -*- coding: utf-8 -*-
import sys
if sys.version_info <= (2, 8):
    from builtins import super

import unittest

import numpy as np
import pandas as pd

import sfa
from sfa import calc_accuracy
import sfa.base
from sfa import AlgorithmSet
from sfa import DataSet

from sfa.data import borisov_2009


class SimpleData(sfa.base.Data):
    def __init__(self):
        super().__init__()
        self._abbr = "SC"
        self._name = "Simple cascade"
        self._A = np.array([[0, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0]], dtype=np.float)

        self._n2i = {"A": 0, "B": 1, "C": 2}
        self._df_ba = pd.DataFrame()
        self._df_exp = pd.DataFrame()

        self._dg = None  # nx.DiGraph()

class TestAlgorithmGS(unittest.TestCase):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create an object for signal propagation algorithm
        self.algs = AlgorithmSet()
        self.algs.create("GS")

        # Create container for data.
        self.ds = DataSet()
        self.ds.create("NELANDER_2008")
        self.ds.create("BORISOV_2009")

        self.solutions = {}

        self.solutions["NELANDER_2008"] = 0.650

        self.solutions["BORISOV_2009_AUC_LOW"] = 0.551
        self.solutions["BORISOV_2009_AUC_EGF"] = 0.581
        self.solutions["BORISOV_2009_AUC_I"] = 0.588
        self.solutions["BORISOV_2009_AUC_EGF+I"] = 0.583

        self.solutions["BORISOV_2009_SS_LOW"] = 0.587
        self.solutions["BORISOV_2009_SS_EGF"] = 0.537
        self.solutions["BORISOV_2009_SS_I"] = 0.557
        self.solutions["BORISOV_2009_SS_EGF+I"] = 0.542
    # end of def __init__

    def test_simple_data_01(self):

        sdata = SimpleData()
        alg = self.algs["GS"]
        alg.data = sdata
        alg.initialize()

        # Test #1
        alg.params.alpha = 0.5
        b = np.array([1.0, 0.0, 0.0])
        x = alg.compute(b)
        self.assertAlmostEqual(x[0], 0.625, 4)
        self.assertAlmostEqual(x[1], 0.25, 4)
        self.assertAlmostEqual(x[2], 0.125, 4)

        # Test #2
        alg.params.alpha = 0.9
        alg.initialize(data=False)
        b = np.array([1.0, 0.0, 0.0])
        x = alg.compute(b)
        self.assertAlmostEqual(x[0], 0.38928571, 4)
        self.assertAlmostEqual(x[1], 0.32142857, 4)
        self.assertAlmostEqual(x[2], 0.28928571, 4)

    def test_simple_data_02(self):

        sdata = SimpleData()

        sdata._A[0, 0] = -1
        sdata._A[1, 1] = -1

        alg = self.algs["GS"]
        alg.data = sdata

        alg.params.initialize()

        # Test #1
        alg.params.alpha = 0.5
        alg.initialize()
        b = np.array([1.0, 0.5, 0.25])
        x = alg.compute(b)
        self.assertAlmostEqual(x[0], 0.1875, 4)
        self.assertAlmostEqual(x[1], 0.125, 4)
        self.assertAlmostEqual(x[2], 0.1875, 4)

        # Test #2
        alg.params.alpha = 0.9
        alg.initialize(data=False)
        b = np.array([1.0, 0.5, 0.25])
        x = alg.compute(b)
        self.assertAlmostEqual(x[0], 0.02572963, 4)
        self.assertAlmostEqual(x[1], 0.02039588, 4)
        self.assertAlmostEqual(x[2], 0.04335629, 4)

    def test_nelander(self):
        alg = self.algs["GS"]
        data = self.ds["NELANDER_2008"]

        alg.data = data
        alg.params.initialize()
        alg.initialize()
        alg.compute_batch()
        acc = calc_accuracy(alg.result.df_sim, data.df_exp)
        self.assertAlmostEqual(acc, self.solutions[data.abbr], 2)


    def test_borisov(self):
        alg = self.algs["GS"]
        borisov = borisov_2009.create_test_data()

        alg.params.initialize()
        alg.params.use_rel_change = True
        for i, (abbr, data) in enumerate(borisov.items()):
            alg.data = data
            if i == 0:
                alg.initialize()
            else:
                alg.initialize(network=False)

            alg.compute_batch()
            acc = calc_accuracy(alg.result.df_sim, data.df_exp)
            self.assertAlmostEqual(acc, self.solutions[abbr], 2)
        # end of for
    # end of def


if __name__ == "__main__":
    unittest.main(verbosity=2)
