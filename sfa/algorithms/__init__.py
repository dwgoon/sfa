# -*- coding: utf-8 -*-


import os
import glob
import importlib
import collections

class Algorithms(collections.MutableMapping):
    """
    An simple singleton object, which handles algorithm objects with
    its dictionary (i.e., mapping) functionality.
    """
    __instance = None
    def __new__(cls):
        if not Algorithms.__instance:
            Algorithms.__instance = object.__new__(cls)            
        return Algorithms.__instance
        
    def __init__(self, *args, **kwargs):
        self._algorithms = dict()
        # Directory path with the file pattern for algorithm modules
        dir_path = __path__[0]            
        dir_fpat = os.path.join(dir_path, "[a-zA-Z0-9]*.py")        
                
        # Import algorithms based on its file name
        for fdir in glob.glob(dir_fpat):
            fname = os.path.basename(fdir)
            alg_mod_name = fname.split('.')[0] # Algorithm module name
            fstr_module_path = "%s.%s"%(__package__,
                                        alg_mod_name)
            
            mod = importlib.import_module(fstr_module_path)
            print(mod.__name__)
            if "this_should_be_imported" in mod.__name__:
                continue
            
            alg = mod.create_algorithm()
            self._algorithms[alg.id] = alg            
        # end of for
        self.update(dict(*args, **kwargs))
        
    def __getitem__(self, key):
        return self._algorithms[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self._algorithms[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self._algorithms[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self._algorithms)

    def __len__(self):
        return len(self._algorithms)

    def __keytransform__(self, key):
        return key
# end of def class
        
        
"""
<References>

[Dictionary functionality]
http://stackoverflow.com/questions/3387691/python-how-to-perfectly-override-a-dict

[Singleton]
http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
"""