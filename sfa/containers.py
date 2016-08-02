# -*- coding: utf-8 -*-


import os
import re
import importlib
import collections

from abc import ABC, abstractmethod

import sfa.algorithms


class Container(ABC, collections.MutableMapping):
    """
    A simple singleton class, which handles multiple objects with
    its hashable functionality (using dictionary).
    """

    __instance = None

    def __new__(cls):
        if not Container.__instance:
            Container.__instance = super().__new__(cls)
        return Container.__instance

    def __init__(self, *args, **kwargs):
        """
        _dict: internal data structure, which is hashable.
        _dpath: Path for the directory containing algorithms or data.
        """
        self._dict = dict()
        self._dpath = None
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


    def load(self, keys=None):
        """
            Load a single or multiple objects according to keys.
            keys: a single string or multiple strings in an iterable object.
                  All related objects are loaded if 'keys' is None.
        """
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
    def _load_all(self):
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


class AlgorithmSet(Container):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        sfa.algorithms.__file__ represents a directory path
        containing sfa.algorithms's init module (__init__.py).
        """
        self._dpath = os.path.basename(sfa.algorithms.__file__)

    # end of def __init__

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

    def _load_all(self):
        """
        Import all algorithms, based on file names
        """
        for fname in os.listdir(self._dpath):
            if re.match(r"[^_]\w+\.py", fname):
                mod_name = fname.split('.')[0]  # Module name
                self._load_single(mod_name)
        # end of for
    # end of def _load_all

# end of class Algorithms

class DataSet(Container):
    """
    The name of this class is similar to that of 'DataSet' in C#.
    The instance of this class handles multiple sfa.base.Data objects.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        sfa.data.__file__ represents a directory path
        containing sfa.data's init module (__init__.py).
        """
        self._dpath = os.path.basename(sfa.data.__file__)

    # end of def __init__



    def _load_single(self, key):
        pass
    # end of def _load_single

    def _load_all(self):
        """
        Import all algorithms, based on file names
        """
        for fname in os.listdir(self._dpath):
            if re.match(r"[^_]\w+\.py", fname):
                mod_name = fname.split('.')[0]  # Module name
                self._load_single(mod_name)
        # end of for
    # end of def _load_all

# end of def class DataSet