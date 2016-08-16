

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from sfa.utils import FrozenClass


class ContainerItem(ABC):
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

    def initialize(self, init_matrix=True, init_data=True):
        if init_matrix:
            self._initialize_matrix()

        if init_data:
            self._initialize_data()

    def _initialize_matrix(self):
        pass

    def _initialize_data(self):
        N = self._P.shape[0]  # Number of state variables
        n2i = self._data.n2i  # Name to index mapper
        df_ba = self._data.df_ba  # Basal activity

        self._b = np.finfo(np.float).eps * np.ones(N)  # self._b = np.zeros(A.shape[0])
        self._ind_ba = []
        self._val_ba = []
        for i, row in enumerate(df_ba.iterrows()):
            row = row[1]
            list_ind = []  # Indices
            list_val = []  # Values
            for target in df_ba.columns[row.nonzero()]:
                list_ind.append(n2i[target])
                list_val.append(row[target])
            # end of for

            self._ind_ba.append(list_ind)
            self._val_ba.append(list_val)
        # end of for

        # For mapping from the indices of adj. matrix to those of DataFrame
        # (arrange the indices of adj. matrix according to df_exp.columns)
        self._iadj_to_idf = [n2i[x] for x in self._data.df_exp.columns]
    # end of _initialize_data

    def compute(self):
        """Algorithm perform the computation with the given data"""
        df_exp = self._data.df_exp  # Result of experiment

        # Simulation result
        sim_result = np.zeros(df_exp.shape, dtype=np.float)

        b = self._b

        if hasattr(self._data, 'inputs'):  # Input condition
            ind_inputs = [self._data.n2i[inp] for inp in self._data.inputs]
            val_inputs = [val for val in self._data.inputs.values()]
            b[ind_inputs] = val_inputs
        # end of if

        if self._params.is_rel_change:
            x_cnt = self.propagate(b)

        # Main loop of the simulation
        for i, ind_ba in enumerate(self._ind_ba):
            ind_ba = self._ind_ba[i]
            b_store = b[ind_ba][:]
            b[ind_ba] = self._val_ba[i]  # Basal activity

            x_exp = self.propagate(b)

            # Result of a single condition
            if self._params.is_rel_change:  # Use relative change
                rel_change = ((x_exp - x_cnt) / np.abs(x_cnt))
                res_single = rel_change[self._iadj_to_idf]
            else:
                res_single = x_exp[self._iadj_to_idf]

            sim_result[i, :] = res_single
            b[ind_ba] = b_store
        # end of for

        df_sim = pd.DataFrame(sim_result,
                              index=df_exp.index,
                              columns=df_exp.columns)

        # Get the result of elements in the columns of df_exp.
        self._result.df_sim = df_sim[df_exp.columns]
    # end of def compute

    @abstractmethod
    def propagate(self):
        raise NotImplementedError("propagate() of Algorithm "
                                  "should be implemented")

        
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
    def A(self): # Adjacency matrix
        return self._A

    @property
    def n2i(self): # Name to index mapping (hashable)
        return self._n2i

    @property
    def dg(self): # Directed graph object of NetworkX
        return self._dg

    @property
    def inputs(self): # Input conditions
        return self._inputs

    @property # DataFrame of basal activity
    def df_ba(self):
        return self._df_ba

    @property # DataFrame of experimental result
    def df_exp(self):
        return self._df_exp

# end of class Data


class Result(FrozenClass):

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