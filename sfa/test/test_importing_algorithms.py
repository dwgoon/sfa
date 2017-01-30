# -*- coding: utf-8 -*-

import sys
if sys.version_info <= (2, 8):
    from builtins import super

import unittest

import os
import sys
import glob
import importlib

import sfa.base
import sfa.algorithms


class TestImportingAlgorithms(unittest.TestCase):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._algorithms = []
        
        # Directory path with the file pattern for algorithm modules
        dir_path = os.path.dirname(sfa.algorithms.__file__)
        dir_fpat = os.path.join(dir_path, "[a-zA-Z0-9]*.py")        
                
        # Import algorithms based on its file name
        for fdir in glob.glob(dir_fpat):
            fname = os.path.basename(fdir)
            alg_mod_name = fname.split('.')[0] # Algorithm module name
            fstr_module_path = "%s.%s"%(sfa.algorithms.__package__,
                                        alg_mod_name)
            
            mod = importlib.import_module(fstr_module_path)
            self._algorithms.append(mod)
        # end of for
    # end of def setUp

    def test_import_module(self):
        # Module path
        #mpath = 'sfa.algorithms.this_should_be_imported'
        #self.assertTrue(mpath in sys.modules)
        
        mpath = 'sfa.algorithms._this_should_not_be_imported'
        self.assertFalse(mpath in sys.modules)
    # end of def test_import
        
    def test_create_algorithm(self):
        for alg_mod in self._algorithms:
            self.assertTrue( hasattr(alg_mod, 'create_algorithm') )
            alg = alg_mod.create_algorithm("TEST")
            self.assertTrue( isinstance(alg, sfa.base.Algorithm) )
        # end of for            
    # end of def test_create_algorithm

    def test_algorithmset_create(self):
        """
        Test AlgorithmSet.create
        """
        algs = sfa.AlgorithmSet()
        algs.create()

        for key, elem in algs.items():
            if isinstance(elem, dict):
                for subkey, subelem in elem.items():
                    self.assertEqual(subkey, subelem.abbr)
            else:
                self.assertEqual(key, elem.abbr)
    # end of def

    def test_algorithmset_singleton(self):
        a1 = sfa.AlgorithmSet()
        a1.create()
        self.assertTrue(len(a1) != 0)

        a2 = sfa.AlgorithmSet()
        a2.create()
        self.assertTrue(len(a1) != 0)
        self.assertEqual(len(a1), len(a2))
        self.assertEqual(a1, a2)

    # end of def





if __name__ == "__main__":
    unittest.main(verbosity=2)
    


    