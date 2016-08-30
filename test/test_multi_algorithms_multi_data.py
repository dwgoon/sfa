# -*- coding: utf-8 -*-

import unittest
from collections import defaultdict
import numpy as np
import pandas as pd

from sfa import calc_accuracy
from sfa import AlgorithmSet
from sfa import DataSet
from sfa.algorithms.sp import SignalPropagation


class TestMultipleAlgorithmsMultipleData(unittest.TestCase):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create an object for signal propagation algorithm
        self.algs = AlgorithmSet()
        self.algs.create()


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

        self.solutions["GS"]["BORISOV_2009_AUC_EGF"] = 0.581
        self.solutions["GS"]["BORISOV_2009_AUC_CTRL"] = 0.551
        self.solutions["GS"]["BORISOV_2009_AUC_I"] = 0.588
        self.solutions["GS"]["BORISOV_2009_AUC_EGF+I"] = 0.583
        self.solutions["GS"]["BORISOV_2009_SS_CTRL"] = 0.587
        self.solutions["GS"]["BORISOV_2009_SS_EGF"] = 0.537
        self.solutions["GS"]["BORISOV_2009_SS_I"] = 0.557
        self.solutions["GS"]["BORISOV_2009_SS_EGF+I"] = 0.542

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
                if isinstance(alg, SignalPropagation):
                    alg.params.is_rel_change = True
                alg.initialize()
                alg.compute_batch()
                acc = calc_accuracy(alg.result.df_sim, data.df_exp)
                res[alg_name][data_name] = acc
            # end of for
        # end of for

        self.assertTrue(np.allclose(pd.DataFrame(res),
                                    pd.DataFrame(self.solutions),
                                    1e-2))

    # end of def test_multiple

# end of def class