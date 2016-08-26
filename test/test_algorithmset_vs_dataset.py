# -*- coding: utf-8 -*-

import unittest

from sfa import calc_accuracy
from sfa import AlgorithmSet
from sfa import DataSet


class TestAlgorithmSP(unittest.TestCase):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create an object for signal propagation algorithm
        self.algs = AlgorithmSet()
        self.algs.create("SP")


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