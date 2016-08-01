
from abc import ABC, abstractmethod


class Result(ABC):
    pass

# end of class Result


class Algorithm(ABC):
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
        self._abbr = abbr
        self._name = None
        self._data = None
        self._params = None
        self._result = None

    def __str__(self):
        return self._name

    # Read-only properties
    @property
    def abbr(self):
        return self._abbr

    @property
    def name(self):
        return self._name

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

    # Methods
    def initialize(self):
        pass
    
    @abstractmethod
    def compute(self):
        """Algorithm perform the computation with the given data"""
    # end of def compute
        
# end of class Algorithm        


class Data(ABC):
    def __init__(self, abbr):
        """
        abbr: Abbreviation of data name
        name: Full name of this data
        """
        self._abbr = abbr
        self._name = None

    def __str__(self):
        return self._name

    # Read-only properties
    @property
    def abbr(self):
        return self._abbr

    @property
    def name(self):
        return self._name

# end of class Data
