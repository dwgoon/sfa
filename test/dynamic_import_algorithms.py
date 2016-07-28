# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:53:27 2016

@author: dwlee
"""


import os
import glob
from importlib import import_module

import sfa.algorithms

if __name__ == "__main__":
    
    dir_path = sfa.algorithms.__path__[0]
    file_names = os.listdir(dir_path)
    
    # Directory path with the file pattern for algorithm modules
    dir_fpat = os.path.join(dir_path, "[a-zA-Z0-9]*.py")

    
    
    for fdir in glob.glob(dir_fpat):
        print(fdir)
        fname = os.path.basename(fdir)
        alg_mod_name = fname.split('.')[0] # Algorithm module name
        fstr_module_path = "%s.%s"%(sfa.algorithms.__package__,
                                    alg_mod_name)
        
        import_module(fstr_module_path)

