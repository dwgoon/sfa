# -*- coding: utf-8 -*-
import sys
if sys.version_info <= (2, 8):
    from builtins import super

import unittest

import os
import importlib

import sfa.base
import sfa.data


class TestImportingData(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ds = sfa.DataSet()
        self.ds.create(["BORISOV_2009", "MOLINELLI_2013"])

    # end of def __init__

    def test_verify_data_object(self):
        for abbr in self.ds:
            if isinstance(self.ds[abbr], dict):
                for subabbr in self.ds[abbr]:
                    data = self.ds[abbr][subabbr]
                    self.assertTrue(isinstance(data, sfa.base.Data))
            else:
                data = self.ds[abbr]
                self.assertTrue(isinstance(data, sfa.base.Data))
        # end of for
    # end of def test_create_data

    def test_data_size(self):
        data = self.ds["BORISOV_2009"]["AUC_EGF=1+I=100"]
        self.assertEqual(data.df_exp.shape, (66, 13))

        data = self.ds["MOLINELLI_2013"]
        self.assertEqual(data.df_exp.shape, (44, 25))
    # end of def test_data_size


    def test_dataset_singleton(self):
        ds1 = sfa.DataSet()
        ds1.create()
        self.assertTrue(len(ds1) != 0)

        ds2 = sfa.DataSet()
        ds2.create()
        self.assertTrue(len(ds1) != 0)
        self.assertEqual(len(ds1), len(ds2))
        self.assertEqual(ds1, ds2)
    # end of def

# end of def class TestImportingData

if __name__ == "__main__":
    unittest.main(verbosity=2)
