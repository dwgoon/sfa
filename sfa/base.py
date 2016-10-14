import sys
if sys.version_info <= (2, 8):
    from builtins import super

# from abc import ABC, abstractmethod
import abc
import copy

import six

import sfa.utils


@six.add_metaclass(abc.ABCMeta)
class ContainerItem():
    def __init__(self, abbr=None, name=None):
        """
        abbr: Abbreviation (or symbol) representing this item
        name: Full name of this item
        """
        self._abbr = abbr
        self._name = name

    def __str__(self):
        return self._abbr

    def __repr__(self):
        class_name = self.__class__.__name__
        return "%s object" % (class_name)

    # Read-only properties
    @property
    def abbr(self):
        return self._abbr

    @property
    def name(self):
        return self._name


class Algorithm(ContainerItem):
    """
    The algorithms should implement compute method.    
    
    Usage:
        >>> class AnAlgorithm(sfa.Algorithm):
                ...
                ...        
        
        >>> alg = AnAlgorithm()
        >>> alg.params = params_obj # Parameters of the algorithm
        >>> alg.data = data_obj # Data to be analyzed by the algorithm
        >>> alg.initialize()
        >>> res = alg.compute()    
        
    """
    def __init__(self, abbr):
        """
        abbr: Abbreviation of algorithm name
        name: Full name of this algorithm
        """
        super().__init__(abbr)
        self._data = None
        self._params = None
        self._result = None

    def copy(self, is_deep=False):
        if is_deep:
            copy.deepcopy(self)
        else:
            return copy.copy(self)

    # Read-only properties
    @property
    def result(self):
        return self._result

    # Read & write properties
    @property
    def params(self):
        return self._params
        
    @params.setter
    def params(self, obj):
        self._params = obj    
    
    @property
    def data(self):
        return self._data
        
    @data.setter
    def data(self, obj):
        self._data = obj

    def initialize(self, network=True, ba=True, data=True):

        if network:
            self._initialize_network()

        if ba:
            self._initialize_basal_activity()

        if data:
            self._initialize_data()

    def _initialize_params(self):
        pass

    def _initialize_network(self):
        pass

    def _initialize_basal_activity(self):
        pass

    def _initialize_data(self):
        pass

    @abc.abstractmethod
    def compute_batch(self):
        raise NotImplementedError("compute() should be implemented")

# end of class Algorithm        


class Data(ContainerItem):
    def __init__(self):
        super().__init__()
        self._A = None
        self._n2i = None
        self._dg = None
        self._inputs= None
        self._df_ba = None
        self._df_exp = None

    # Read-only members
    @property
    def A(self):  # Adjacency matrix
        return self._A

    @property
    def n2i(self):  # Name to index mapping (hashable)
        return self._n2i

    @property
    def dg(self):  # Directed graph object of NetworkX
        return self._dg

    @property
    def inputs(self):  # Input conditions
        return self._inputs

    @property  # DataFrame of basal activity
    def df_ba(self):
        return self._df_ba

    @property  # DataFrame of experimental result
    def df_exp(self):
        return self._df_exp

# end of class Data


class Result(sfa.utils.FrozenClass):

    def __init__(self):
        self._df_sim = None
        self._freeze()

    @property
    def df_sim(self):
        return self._df_sim

    @df_sim.setter
    def df_sim(self, val):
        self._df_sim = val

# end of def class Result
