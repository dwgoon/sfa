# -*- coding: utf-8 -*-

import unittest

import os
import glob
import importlib

import sfa.base
import sfa.data


class TestImportingData(unittest.TestCase):

    def setUp(self):
        self._data = {}
        
        # Directory path with the file pattern for algorithm modules
        dir_path = os.path.dirname(sfa.data.__file__)
                
        # Create all data objects
        for elem in os.listdir(dir_path):
            dpath = os.path.join(dir_path, elem)
            if not elem.startswith('_') and os.path.isdir(dpath):
                fstr_module_path = "%s.%s"%(sfa.data.__package__,
                                            elem)
            
                mod = importlib.import_module(fstr_module_path)
                data = mod.create_data()
                if type(data) is dict:
                    self._data.update(data)
                elif type(data) is list:
                    for obj in data:
                        self._data[obj.abbr] = obj
                elif isinstance(data, sfa.base.Data):
                    self._data[data.abbr] = data
                else:
                    self.fail("%s.create_data() returns unsupported type."%(dname))
            # end of if
        # end of for
    # end of def setUp

    def test_verify_data_object(self):
        for abbr, data in self._data.items():
            self.assertTrue(isinstance(data, sfa.base.Data))
        # end of for
    # end of def test_create_data

    def test_data_size(self):
        data = self._data["BORISOV_2009_AUC_EGF+I"]
        self.assertEqual(data.df_exp.shape, (91, 13))

        data = self._data["MOLINELLI_2013"]
        self.assertEqual(data.df_exp.shape, (44, 25))
    # end of def test_data_size


    def test_dataset_load(self):
        """
        Test DataSet.load
        """
        ds = sfa.DataSet()
        data = ds.load("NELENDER_2008")  # Single algorithm
        self.assertTrue(len(algs) == 1)

        algs.load(["GS", ])  # Algorithms in an iterable object
        self.assertTrue(len(algs) == 2)

        algs.load()  # Load all algorithms
        self.assertTrue(len(algs) == (len(self._algorithms) - 1))

if __name__ == "__main__":
    unittest.main(verbosity=2)
    


    