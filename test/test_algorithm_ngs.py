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
        self._df_conds = pd.DataFrame()
        self._df_exp = pd.DataFrame()

        self._dg = None  # nx.DiGraph()

class TestAlgorithmNGS(unittest.TestCase):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create an object for signal propagation algorithm
        self.algs = AlgorithmSet()
        self.algs.create("NGS")

    # end of def __init__

    def test_simple_data_01(self):
        """
        A simple cascade
        (A) -> (B) -> (C)
        """

        sdata = SimpleData()

        alg = self.algs["NGS"]
        alg.data = sdata

        alg.params.initialize()

        # Test #1
        alg.params.alpha = 0.5
        alg.initialize()
        b = np.array([1.0, 0., 0.])
        x = alg.compute(b)
        self.assertAlmostEqual(x[0], 0.625, 4)
        self.assertAlmostEqual(x[1], 0.25, 4)
        self.assertAlmostEqual(x[2], 0.125, 4)

        # Test #2
        alg.params.alpha = 0.9
        alg.initialize(data=False)
        b = np.array([1.0, 0., 0.])
        x = alg.compute(b)
        self.assertAlmostEqual(x[0], 0.38928571, 4)
        self.assertAlmostEqual(x[1], 0.32142857, 4)
        self.assertAlmostEqual(x[2], 0.28928571, 4)

    # end of def

    def test_simple_data_02(self):
        """
        Symmetric two-link 3-node network
        (A) -> (B)
        (A) -> (C)
        """

        sdata = SimpleData()
        sdata._A[2, 1] = 0
        sdata._A[2, 0] = 1

        alg = self.algs["NGS"]
        alg.data = sdata

        alg.params.initialize()

        # Test #1
        alg.params.alpha = 0.5
        alg.initialize()
        b = np.array([1.0, 0., 0.])
        x = alg.compute(b)
        self.assertAlmostEqual(x[0], 0.66666667, 4)
        self.assertAlmostEqual(x[1], 0.23570226, 4)
        self.assertAlmostEqual(x[2], 0.23570226, 4)

        # Test #2
        alg.params.alpha = 0.9
        alg.initialize(data=False)
        b = np.array([1.0, 0., 0.])
        x = alg.compute(b)
        self.assertAlmostEqual(x[0], 0.52631579, 4)
        self.assertAlmostEqual(x[1], 0.33494532, 4)
        self.assertAlmostEqual(x[2], 0.33494532, 4)

    # end of def


    def test_simple_data_03(self):
        """
        A simple cascade with self-negative link of (B)
        (A) -> (B) -> (C)
        (B) -| (B)
        """

        sdata = SimpleData()
        sdata._A[1, 1] = -1

        alg = self.algs["NGS"]
        alg.data = sdata

        alg.params.initialize()

        # Test #1
        alg.params.alpha = 0.5
        alg.initialize()
        b = np.array([1.0, 0., 0.])
        x = alg.compute(b)
        self.assertAlmostEqual(x[0], 0.5625, 4)
        self.assertAlmostEqual(x[1], 0.125, 4)
        self.assertAlmostEqual(x[2], 0.0625, 4)

        # Test #2
        alg.params.alpha = 0.9
        alg.initialize(data=False)
        b = np.array([1.0, 0., 0.])
        x = alg.compute(b)
        self.assertAlmostEqual(x[0], 0.13894231, 4)
        self.assertAlmostEqual(x[1], 0.04326923, 4)
        self.assertAlmostEqual(x[2], 0.03894231, 4)

    # end of def

    def test_simple_data_04(self):
        """
        A simple cascade with self-negative link of (B)
        (A) -> (B) -> (C)
        (B) -| (B)
        """

        sdata = SimpleData()
        sdata._A[1, 1] = -1

        alg = self.algs["NGS"]
        alg.data = sdata

        alg.params.initialize()

        # Test #1
        alg.params.alpha = 0.5
        alg.initialize()
        b = np.array([1.0, 0., 0.])
        x = alg.compute(b)
        self.assertAlmostEqual(x[0], 0.5625, 4)
        self.assertAlmostEqual(x[1], 0.125, 4)
        self.assertAlmostEqual(x[2], 0.0625, 4)

        # Test #2
        alg.params.alpha = 0.9
        alg.initialize(data=False)
        b = np.array([1.0, 0., 0.])
        x = alg.compute(b)
        self.assertAlmostEqual(x[0], 0.13894231, 4)
        self.assertAlmostEqual(x[1], 0.04326923, 4)
        self.assertAlmostEqual(x[2], 0.03894231, 4)


    # end of def

if __name__ == "__main__":
    unittest.main(verbosity=2)
