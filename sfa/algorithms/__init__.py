# -*- coding: utf-8 -*-


import os
import re
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
            Algorithms.__instance = super().__new__(cls)            
        return Algorithms.__instance
        
    def __init__(self, *args, **kwargs):
        self._algorithms = dict()
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

    def _load_algorithm(self, abbr):
        abbr_low = abbr.lower()
        fstr_module_path = "%s.%s"%(__package__,
                                    abbr_low)
        mod = importlib.import_module(fstr_module_path)
        abbr_up = abbr.upper()
        if abbr_up in self._algorithms:
            return
        elif "this_should_be_imported" in mod.__name__:
            return

        alg = mod.create_algorithm(abbr_up)
        self._algorithms[abbr_up] = alg

        # For testing purpose
        print( "%s has been loaded."%(mod.__name__) )


    def load(self, abbr=None):
        """
        abbr: a single abbreviation in string or multiple abbreviations in
              an iterable object. All algorithms are loaded if abbr is None.
        """
        dir_path = __path__[0]
        if abbr is not None:
            if type(abbr) is str:
                self._load_algorithm(abbr)
            elif hasattr(abbr, '__iter__'):
            # An iterable object contains multiple abbreviations of algorithms.
                for elem in abbr:
                    self._load_algorithm(elem)
        else:       
            # Import algorithms based on its file name
            for fname in os.listdir(dir_path):
                if re.match(r"[^_]\w+\.py", fname):                    
                    alg_mod_name = fname.split('.')[0] # Algorithm module name
                    self._load_algorithm(alg_mod_name)
            # end of for

# end of def class
        
        
"""
<References>

[Dictionary functionality]
http://stackoverflow.com/questions/3387691/python-how-to-perfectly-override-a-dict

[Singleton]
http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
"""