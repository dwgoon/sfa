# -*- coding: utf-8 -*-


import os
import re
import importlib
import collections

from abc import ABC, abstractmethod

import sfa.algorithms
import sfa.data


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
        self._map = dict()
        self._dpath = None
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self._map[key]

    def __setitem__(self, key, value):
        self._map[key] = value

    def __delitem__(self, key):
        del self._map[key]

    def __iter__(self):
        return iter(self._map)

    def __len__(self):
        return len(self._map)

    def keys(self):
        return self._map.keys()

    def values(self):
        return self._map.values()

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
        self._dpath = os.path.dirname(sfa.algorithms.__file__)

    # end of def __init__

    def _load_single(self, key):
        key_low = key.lower()
        fstr_module_path = "%s.%s" % (sfa.algorithms.__package__,
                                      key_low)

        key_up = key.upper()  # Avoid redundant importing
        if key_up in self._map:
            return

        mod = importlib.import_module(fstr_module_path)
        # The following is of test purpose (removable in the future).
        if "this_should_be_imported" in mod.__name__:
            return

        alg = mod.create_algorithm(key_up)
        self._map[key_up] = alg

        # For testing purpose
        print("%s has been loaded." % (mod.__name__))

    def _load_all(self):
        """
        Import all algorithms, based on file names
        """
        for entity in os.listdir(self._dpath):
            if re.match(r"[^_]\w+\.py", entity):
                mod_name = entity.split('.')[0]  # Module name
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
        self._dpath = os.path.dirname(sfa.data.__file__)

    # end of def __init__

    def _load_single(self, key):
        key_items = key.split("_")
        key_1st, key_2nd = key_items[:2]
        mod_name = "%s_%s"%(key_1st.lower(), key_2nd.lower())
        fstr_module_path = "%s.%s" % (sfa.data.__package__, mod_name)

        key_upper = key.upper()
        if key_upper in self._map:  # Avoid redundant importing
            return
        elif mod_name.upper() in self._map:
            # All data object in this directory has been created before.
            return

        mod = importlib.import_module(fstr_module_path)
        if len(key_items) > 2:  # Create the specified single data object
            data = mod.create_data(key)
        else:  # Create all data objects from this directory
            data = mod.create_data()

        if type(data) is dict:
            self._map[key_upper] = data
        elif type(data) is list:
            self._map[key_upper] = {obj.abbr: obj for obj in data}
        elif isinstance(data, sfa.base.Data):
            self._map[data.abbr] = data
        else:
            err_msg = "%s.create_data() returns unsupported type."\
                      % (fstr_module_path)
            raise TypeError(err_msg)
        # end of if
    # end of def _load_single

    def _load_all(self):
        """
        Import all data, based on the directory names of data modules
        """
        for entity in os.listdir(self._dpath):
            dpath = os.path.join(self._dpath, entity)
            if not entity.startswith('_') and os.path.isdir(dpath):
                self._load_single(entity)
        # end of for
    # end of def _load_all

# end of def class DataSet
