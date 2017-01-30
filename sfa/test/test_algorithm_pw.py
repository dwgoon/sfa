# -*- coding: utf-8 -*-

import sys
if sys.version_info <= (2, 8):
    from builtins import super

import unittest

import sfa
from sfa import calc_accuracy
from sfa import AlgorithmSet
from sfa import DataSet
from sfa.data import borisov_2009


class TestAlgorithmAPS(unittest.TestCase):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create an object for signal propagation algorithm
        self.algs = AlgorithmSet()
        self.algs.create("APS")

        # Create container for data.
        self.ds = DataSet()
        self.ds.create("NELANDER_2008")
        self.ds.create("BORISOV_2009")

        self.solutions = {}

        self.solutions["NELANDER_2008"] = 0.751

        self.solutions["BORISOV_2009_AUC_LOW"] = 0.682
        self.solutions["BORISOV_2009_AUC_EGF"] = 0.725
        self.solutions["BORISOV_2009_AUC_I"] = 0.770
        self.solutions["BORISOV_2009_AUC_EGF+I"] = 0.742

        self.solutions["BORISOV_2009_SS_LOW"] = 0.704
        self.solutions["BORISOV_2009_SS_EGF"] = 0.664
        self.solutions["BORISOV_2009_SS_I"] = 0.697
        self.solutions["BORISOV_2009_SS_EGF+I"] = 0.650
    # end of def __init__

    def test_nelander(self):
        alg = self.algs["APS"]
        data = self.ds["NELANDER_2008"]

        alg.params.initialize()
        alg.params.no_inputs = True
        alg.data = data
        alg.initialize()
        alg.compute_batch()
        acc = calc_accuracy(alg.result.df_sim, data.df_exp)
        self.assertAlmostEqual(acc, self.solutions[data.abbr], 2)

    def test_borisov(self):
        alg = self.algs["APS"]
        borisov = borisov_2009.create_test_data()

        alg.params.initialize()
        alg.params.no_inputs = True
        alg.data = alg.data = sfa.get_avalue(borisov)
        alg.initialize(data=False)
        for abbr, data in borisov.items():
            alg.data = data
            alg.initialize(network=False)
            alg.compute_batch()
            acc = calc_accuracy(alg.result.df_sim, data.df_exp)
            #print (abbr, acc)
            self.assertAlmostEqual(acc, self.solutions[abbr], 2)
        # end of for
    # end of def


if __name__ == "__main__":
    unittest.main(verbosity=2)
