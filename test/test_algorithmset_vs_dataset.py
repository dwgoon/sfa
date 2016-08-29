# -*- coding: utf-8 -*-

import unittest
from collections import defaultdict
import numpy as np

from sfa import calc_accuracy
from sfa import AlgorithmSet
from sfa import DataSet


class TestMultipleAlgorithmsMultipleData(unittest.TestCase):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create an object for signal propagation algorithm
        self.algs = AlgorithmSet()
        self.algs.create("SP")


        # Create container for data.
        self.ds = DataSet()
        self.ds.create("BORISOV_2009")

        self.solutions = defaultdict(dict)

        self.solutions["SP"]["BORISOV_2009_AUC_CTRL"] = 0.69822
        self.solutions["SP"]["BORISOV_2009_AUC_EGF"] = 0.70583
        self.solutions["SP"]["BORISOV_2009_AUC_I"] = 0.79205
        self.solutions["SP"]["BORISOV_2009_AUC_EGF+I"] = 0.77430
        self.solutions["SP"]["BORISOV_2009_SS_CTRL"] = 0.72527
        self.solutions["SP"]["BORISOV_2009_SS_EGF"] = 0.65765
        self.solutions["SP"]["BORISOV_2009_SS_I"] = 0.73119
        self.solutions["SP"]["BORISOV_2009_SS_EGF+I"] = 0.66272

        self.solutions["GS"]["BORISOV_2009_AUC_CTRL"] = 0.69822
        self.solutions["GS"]["BORISOV_2009_AUC_EGF"] = 0.70583
        self.solutions["GS"]["BORISOV_2009_AUC_I"] = 0.79205
        self.solutions["GS"]["BORISOV_2009_AUC_EGF+I"] = 0.77430
        self.solutions["GS"]["BORISOV_2009_SS_CTRL"] = 0.72527
        self.solutions["GS"]["BORISOV_2009_SS_EGF"] = 0.65765
        self.solutions["GS"]["BORISOV_2009_SS_I"] = 0.73119
        self.solutions["GS"]["BORISOV_2009_SS_EGF+I"] = 0.66272

        self.solutions["PW"]["BORISOV_2009_AUC_CTRL"] = 0.648
        self.solutions["PW"]["BORISOV_2009_AUC_EGF"] = 0.689
        self.solutions["PW"]["BORISOV_2009_AUC_I"] = 0.732
        self.solutions["PW"]["BORISOV_2009_AUC_EGF+I"] = 0.704
        self.solutions["PW"]["BORISOV_2009_SS_CTRL"] = 0.669
        self.solutions["PW"]["BORISOV_2009_SS_EGF"] = 0.647
        self.solutions["PW"]["BORISOV_2009_SS_I"] = 0.663
        self.solutions["PW"]["BORISOV_2009_SS_EGF+I"] = 0.638
    # end of def __init__

    def test_multiple(self):
        algs = self.algs
        ds = self.ds

        res = defaultdict(dict) # np.zeros((len(algs), len(ds)), dtype=np.float)

        for alg_name, alg in algs.items():
            for data_name, data in ds["BORISOV_2009"].items():
                alg.data = data
                alg.initialize()
                alg.compute_panel()
                acc = calc_accuracy(alg.result.df_sim, data.df_exp)
                res[data_name][alg_name] = acc
            # end of for
        # end of for



    # end of def test_multiple

# end of def class