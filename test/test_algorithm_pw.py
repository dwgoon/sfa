# -*- coding: utf-8 -*-

import unittest

from sfa import calc_accuracy
from sfa import AlgorithmSet
from sfa import DataSet


class TestAlgorithmPW(unittest.TestCase):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create an object for signal propagation algorithm
        self.algs = AlgorithmSet()
        self.algs.create("PW")

        # Create container for data.
        self.ds = DataSet()
        self.ds.create("NELENDER_2008")
        self.ds.create("BORISOV_2009")

        self.solutions = {}

        self.solutions["NELENDER_2008"] = 0.798

        self.solutions["BORISOV_2009_AUC_CTRL"] = 0.648
        self.solutions["BORISOV_2009_AUC_EGF"] = 0.689
        self.solutions["BORISOV_2009_AUC_I"] = 0.732
        self.solutions["BORISOV_2009_AUC_EGF+I"] = 0.704

        self.solutions["BORISOV_2009_SS_CTRL"] = 0.669
        self.solutions["BORISOV_2009_SS_EGF"] = 0.647
        self.solutions["BORISOV_2009_SS_I"] = 0.663
        self.solutions["BORISOV_2009_SS_EGF+I"] = 0.638
    # end of def

    def test_sp_nelender(self):
        alg = self.algs["PW"]
        data = self.ds["NELENDER_2008"]

        alg.params.initialize()
        alg.data = data
        alg.initialize()
        alg.compute()
        acc = calc_accuracy(alg.result.df_sim, data.df_exp)
        self.assertAlmostEqual(acc, self.solutions[data.abbr], 2)

    def test_sp_borisov(self):
        alg = self.algs["PW"]
        borisov = self.ds["BORISOV_2009"]

        alg.params.initialize()
        #alg.params.is_rel_change = True
        alg.data = borisov["BORISOV_2009_AUC_CTRL"]
        alg.initialize(init_data=False)
        for abbr, data in borisov.items():
            alg.data = data
            alg.initialize(init_network=False)
            alg.compute()
            acc = calc_accuracy(alg.result.df_sim, data.df_exp)
            self.assertAlmostEqual(acc, self.solutions[abbr], 2)
        # end of for
    # end of def


if __name__ == "__main__":
    unittest.main(verbosity=2)
