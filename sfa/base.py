
from abc import ABC, abstractmethod


class Data(ABC):
    pass       
    
# end of class Data
    

class Result(ABC):
    pass
    
    
# end of class Result
    

class Algorithm(ABC):    
    def __init__(self):
        self._id = None
        self._name = None
        self._data = None

    def __str__(self):
        return self._name

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name
    
    @property
    def data(self):
        return self._data
        
    @data.setter
    def data(self, obj):
        self._data = obj

    def initialize(self):
        pass
    
    @abstractmethod
    def compute(self):
        """Algorithm perform the computation with the given data"""
    # end of def compute
        
        
# end of class Algorithm        
