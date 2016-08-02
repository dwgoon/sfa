# -*- coding: utf-8 -*-


import os
import re
import importlib
import collections

from abc import ABC, abstractmethod

import sfa.algorithms


class SingletonContainer(ABC, collections.MutableMapping):
    """
    An simple singleton class, which handles multiple objects with
    its hashable functionality (using dictionary).
    """

    __instance = None

    def __new__(cls):
        if not SingletonContainer.__instance:
            SingletonContainer.__instance = super().__new__(cls)
        return SingletonContainer.__instance

    def __init__(self, *args, **kwargs):
        self._dict = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self._dict[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self._dict[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self._dict[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __keytransform__(self, key):
        return key

    @abstractmethod
    def load(self, keys=None):
        """
        Load a single or multiple objects according to keys.
        keys: a single string or multiple strings in an iterable object.
              All related objects are loaded if 'keys' is None.
        """
        # dir_path should be defined according to the class
        # self._load(dir_path, keys)
    # end of load

    def load(self, fpath, keys=None):
        dir_path = os.path.basename(fpath)
        if keys is not None:
            if type(keys) is str:
                self._load_single(keys)
            elif hasattr(keys, '__iter__'):
                # An iterable object contains multiple keys.
                for elem in keys:
                    self._load_single(elem)
        else:
            self._load_all()
    # end of def _load

    @abstractmethod
    def _load_single(self, key):
        """Load a single object"""
    # end of def

    @abstractmethod
    def _load_all(self, dir_path):
        """Load all objects"""
    # end of def

"""
<References>

[Dictionary functionality]
http://stackoverflow.com/questions/3387691/python-how-to-perfectly-override-a-dict

[Singleton]
http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
"""
# end of def class


class AlgorithmSet(SingletonContainer):
    def load(self, keys=None):
        # sfa.algorithms.__file__ represents a directory path containing
        # sfa.algorithms.__init__.py
        dir_path = os.path.dirname(sfa.algorithms.__file__)
        self._load(dir_path, keys)
    # end of def load

    def _load_single(self, key):
        key_low = key.lower()
        fstr_module_path = "%s.%s" % (sfa.algorithms.__package__,
                                      key_low)
        mod = importlib.import_module(fstr_module_path)
        key_up = key.upper()
        if key_up in self._dict:
            return
        elif "this_should_be_imported" in mod.__name__:
            return

        alg = mod.create_algorithm(key_up)
        self._dict[key_up] = alg

        # For testing purpose
        print("%s has been loaded." % (mod.__name__))

    def _load_all(self, dir_path):
        """
        Import all algorithms, based on file names
        """
        for fname in os.listdir(dir_path):
            if re.match(r"[^_]\w+\.py", fname):
                mod_name = fname.split('.')[0]  # Module name
                self._load_single(mod_name)
        # end of for
    # end of def _load_all

# end of class Algorithms

class DataSet(SingletonContainer):
    """
    The name of this class is similar to that of 'DataSet' in C#.
    The instance of this class handles multiple sfa.base.Data objects.
    """
    def load(self, keys=None):
        dir_path = os.path.dirname(sfa.data.__file__)
        if keys is not None:
            if type(keys) is str:
                self._load_single(keys)
            elif hasattr(keys, '__iter__'):
                # An iterable object contains multiple abbreviations.
                for elem in keys:
                    self._load_single(elem)
        else:
            # Import algorithms based on its file name
            for fname in os.listdir(dir_path):
                if re.match(r"[^_]\w+\.py", fname):
                    alg_mod_name = fname.split('.')[0]  # Algorithm module name
                    self._load_single(alg_mod_name)
            # end of for

    # end of def

    def _load_data(self, key):
        pass


    def _load_all(self):