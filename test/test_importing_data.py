# -*- coding: utf-8 -*-

import unittest

import os
import sys
import glob
import importlib

import sfa.base
import sfa.algorithms


class TestImportingData(unittest.TestCase):

    def setUp(self):
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
        mpath = 'sfa.algorithms.this_should_be_imported'         
        self.assertTrue(mpath in sys.modules)
        
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

    def test_algorithmset_load(self):
        """
        Test AlgorithmSet.load
        """
        algs = sfa.AlgorithmSet()
        algs.load("SP") # Single algorithm
        self.assertTrue( len(algs) == 1 )

        algs.load(["GS",]) # Algorithms in an iterable object
        self.assertTrue( len(algs) == 2 )

        algs.load() # Load all algorithms
        self.assertTrue( len(algs) == (len(self._algorithms)-1)  )

if __name__ == "__main__":
    unittest.main(verbosity=2)
    


    