# -*- coding: utf-8 -*-

import unittest

import numpy as np
import pandas as pd

import sfa
from sfa import calc_accuracy
import sfa.base
from sfa import AlgorithmSet
from sfa import DataSet


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

class TestAlgorithmSP(unittest.TestCase):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create an object for signal propagation algorithm
        self.algs = AlgorithmSet()
        self.algs.create("GS")

        # Create container for data.
        self.ds = DataSet()
        self.ds.create("NELENDER_2008")
        self.ds.create("BORISOV_2009")

        self.solutions = {}

        self.solutions["NELENDER_2008"] = 0.77249

        self.solutions["BORISOV_2009_AUC_CTRL"] = 0.69822
        self.solutions["BORISOV_2009_AUC_EGF"] = 0.70583
        self.solutions["BORISOV_2009_AUC_I"] = 0.79205
        self.solutions["BORISOV_2009_AUC_EGF+I"] = 0.77430

        self.solutions["BORISOV_2009_SS_CTRL"] = 0.72527
        self.solutions["BORISOV_2009_SS_EGF"] = 0.65765
        self.solutions["BORISOV_2009_SS_I"] = 0.73119
        self.solutions["BORISOV_2009_SS_EGF+I"] = 0.66272
    # end of def

    def test_gs_simple_data_01(self):

        sdata = SimpleData()
        alg = self.algs["GS"]
        alg.data = sdata
        alg.initialize()

        # Test #1
        alg.params.alpha = 0.5
        b = np.array([1.0, 0.0, 0.0])
        x = alg.propagate(b)
        self.assertAlmostEqual(x[0], 0.625)
        self.assertAlmostEqual(x[1], 0.25)
        self.assertAlmostEqual(x[2], 0.125)

        # Test #2
        alg.params.alpha = 0.9
        alg.initialize(init_data=False)
        b = np.array([1.0, 0.0, 0.0])
        x = alg.propagate(b)
        self.assertAlmostEqual(x[0], 0.38928571)
        self.assertAlmostEqual(x[1], 0.32142857)
        self.assertAlmostEqual(x[2], 0.28928571)

    def test_gs_simple_data_02(self):

        sdata = SimpleData()

        sdata._A[0, 0] = -1
        sdata._A[1, 1] = -1

        alg = self.algs["GS"]
        alg.data = sdata

        # Test #1
        alg.params.alpha = 0.5
        alg.initialize()
        b = np.array([1.0, 0.5, 0.25])
        x = alg.propagate(b)
        self.assertAlmostEqual(x[0], 0.1875)
        self.assertAlmostEqual(x[1], 0.125)
        self.assertAlmostEqual(x[2], 0.1875)

        # Test #2
        alg.params.alpha = 0.9
        alg.initialize(init_data=False)
        b = np.array([1.0, 0.5, 0.25])
        x = alg.propagate(b)
        self.assertAlmostEqual(x[0], 0.02572963)
        self.assertAlmostEqual(x[1], 0.02039588)
        self.assertAlmostEqual(x[2], 0.04335629)

    def test_gs_nelender(self):
        alg = self.algs["GS"]
        data = self.ds["NELENDER_2008"]

        alg.data = data
        alg.initialize()
        alg.compute()
        acc = calc_accuracy(alg.result.df_sim, data.df_exp)
        # self.assertAlmostEqual(acc, self.solutions[data.abbr], 2)
        print("[GS NELENDER] acc: ", acc)

    def test_gs_borisov(self):
        alg = self.algs["GS"]
        borisov = self.ds["BORISOV_2009"]

        alg.params.is_rel_change = True
        for i, (abbr, data) in enumerate(borisov.items()):
            alg.data = data
            if i == 0:
                alg.initialize()
            else:
                alg.initialize(init_network=False)

            alg.compute()
            acc = calc_accuracy(alg.result.df_sim, data.df_exp)
            #self.assertAlmostEqual(acc, self.solutions[abbr], 2)
            print("[GS %s] acc: %f"%(abbr, acc))

        # end of for
    # end of def


if __name__ == "__main__":
    unittest.main(verbosity=2)
