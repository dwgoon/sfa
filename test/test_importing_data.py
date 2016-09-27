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
                    self.fail("%s.create_data() returns unsupported type." \
                              % (fstr_module_path))
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

    def test_dataset_create(self):
        """
        Test DataSet.create
        """
        ds = sfa.DataSet()
        ds.create()

        for key, elem in ds.items():
            if isinstance(elem, dict):
                for subkey, subelem in elem.items():
                    self.assertEqual(subkey, subelem.abbr)
            else:
                self.assertEqual(key, elem.abbr)

    # end of def test_dataset_create

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
