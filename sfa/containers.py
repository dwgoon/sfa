# -*- coding: utf-8 -*-

import sys
if sys.version_info <= (2, 8):
    from builtins import super

import os
import re
import importlib
import collections
import abc
import six

import sys

if sys.version_info[:2] >= (3, 8):
    from collections.abc import MutableMapping
else:
    from collections import MutableMapping

import sfa.base
import sfa.algorithms
import sfa.data

from sfa.utils import Singleton

__all__ = ["AlgorithmSet", "DataSet"]


@six.add_metaclass(abc.ABCMeta)
class Container(MutableMapping):
    """
    A simple container class, which handles multiple objects with
    its hashable functionality (using dictionary).
    """
    def __init__(self, *args, **kwargs):
        """
        _dict: internal data structure, which is hashable.
        _dpath: Path of the directory containing algorithms, data, etc.
        """
        self._map = dict()
        self._dpath = None
        self._all_keys = None
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self._map[key]

    def __setitem__(self, key, value):
        if isinstance(value, sfa.base.ContainerItem):
            self._map[key] = value
        else:
            raise TypeError("Container.__setitem__ only accepts "
                            "sfa.base.ContainerItem type object.")

    def __delitem__(self, key):
        del self._map[key]

    def __iter__(self):
        return iter(self._map)

    def __len__(self):
        return len(self._map)

    def __str__(self):
        return str(self._map)

    def keys(self):
        return self._map.keys()

    def values(self):
        return self._map.values()

    def create(self, keys=None):
        """
            Create a single or multiple objects according to keys.
            keys: a single string or multiple strings in an iterable object.
                  All related objects are created if 'keys' is None.
        """
        if keys is not None:
            if type(keys) is str:
                return self._create_single(keys)
            elif hasattr(keys, '__iter__'):
                mult = []  # To return multiple elements
                # An iterable object contains multiple keys.
                for elem in keys:
                    mult.append(self._create_single(elem))
                return mult
        else:
            self._create_all()
            return self
    # end of def create

    @abc.abstractmethod
    def get_all_keys(self):
        """"""
    # end of def

    @abc.abstractmethod
    def _create_single(self, key):
        """Create a single object"""
    # end of def

    @abc.abstractmethod
    def _create_all(self):
        """Create all objects"""
    # end of def



"""
<References>

[Dictionary functionality]
http://stackoverflow.com/questions/3387691/python-how-to-perfectly-override-a-dict
"""
# end of def class

excluded = ['np.py']

@Singleton
class AlgorithmSet(Container):
    _instance = None
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        sfa.algorithms.__file__ represents a directory path
        containing sfa.algorithms's init module (__init__.py).
        """
        self._dpath = os.path.dirname(sfa.algorithms.__file__)

    # end of def __init__

    def get_all_keys(self):
        if not self._all_keys:
            self._all_keys = []
            for entity in os.listdir(self._dpath):
                if re.match(r"[^_]\w+\.py$", entity) \
                   and entity not in excluded:
                    mod_name = entity.split('.')[0]  # Module name
                    print(mod_name, entity)
                    key = mod_name.upper()
                    self._all_keys.append(key)
                    yield key
            # end of for
        else:
            return iter(self._all_keys)

    def _create_single(self, key):
        if key in self._map:
            return self._map[key]


        key_low = key.lower()
        fstr_module_path = "%s.%s" % (sfa.algorithms.__name__,
                                      key_low)

        _key = key.upper()  # We use captital characters for the key.
        if _key in self._map:  # Avoid redundant importing
            return

        mod = importlib.import_module(fstr_module_path)
        # The following is of test purpose (removable in the future).
        if "this_should_be_imported" in mod.__name__:
            return

        alg = mod.create_algorithm(_key)
        self._map[_key] = alg

        # For testing purpose
        print("%s algorithm has been created." % (_key))
        return alg

    def _create_all(self):
        """
        Import all algorithms, based on file names
        """
        for entity in os.listdir(self._dpath):
            if re.match(r"[^_]\w+\.py", entity) \
               and entity not in excluded:
                mod_name = entity.split('.')[0]  # Module name
                self._create_single(mod_name)
        # end of for
    # end of def _create_all

# end of class Algorithms

@Singleton
class DataSet(Container):
    """
    The name of this class is similar to that of 'DataSet' in C#.
    The instance of this class handles multiple sfa.base.Data objects.
    """
    _instance = None
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        sfa.data.__file__ represents a directory path
        containing sfa.data's init module (__init__.py).
        """
        self._dpath = os.path.dirname(sfa.data.__file__)

    # end of def __init__

    def get_all_keys(self):
        if not self._all_keys:
            self._all_keys = []
            for entity in os.listdir(self._dpath):
                dpath = os.path.join(self._dpath, entity)
                if not entity.startswith('_') and os.path.isdir(dpath):
                    key = entity.upper()
                    self._all_keys.append(key)
                    yield key
        else:
            return iter(self._all_keys)

    def _create_single(self, key):
        if key in self._map:
            return self._map[key]

        key_items = key.split("_")
        key_1st, key_2nd = key_items[:2]
        mod_name = "%s_%s"%(key_1st.lower(), key_2nd.lower())
        fstr_module_path = "%s.%s" % (sfa.data.__name__, mod_name)

        _key = key.upper()
        if _key in self._map:  # Avoid redundant importing
            return
        elif mod_name.upper() in self._map:
            # All data object in this directory has been created before.
            return

        mod = importlib.import_module(fstr_module_path)
        if len(key_items) > 2:  # Create the specified single data object
            data = mod.create_data(key)
        else:  # Create all data objects from this directory
            data = mod.create_data()

        if isinstance(data, dict):
            self._map[_key] = data
        elif isinstance(data, list):
            self._map[_key] = {obj.abbr.upper(): obj for obj in data}
        elif isinstance(data, sfa.base.Data):
            _key = data.abbr.upper()
            self._map[_key] = data
        else:
            err_msg = "%s.create_data() returns unsupported type."\
                      % (fstr_module_path)
            raise TypeError(err_msg)
        # end of if

        # For testing purpose
        print("%s data has been created." % (_key))
        return self._map[_key]
    # end of def _create_single

    def _create_all(self):
        """
        Import all data, based on the directory names of data modules
        """
        for entity in os.listdir(self._dpath):
            dpath = os.path.join(self._dpath, entity)
            if not entity.startswith('_') and os.path.isdir(dpath):
                self._create_single(entity)
        # end of for
    # end of def _create_all

# end of def class DataSet
