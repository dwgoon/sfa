import os
import sys
if sys.version_info <= (2, 8):
    from builtins import super

# from abc import ABC, abstractmethod
import abc
import copy

import pandas as pd
import six
import sfa.utils

__all__ = ['Algorithm', 'Data', 'Result']

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

    @property
    def abbr(self):
        return self._abbr

    @abbr.setter
    def abbr(self, val):
        self._abbr =val

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = val


class Algorithm(ContainerItem):
    """The algorithms should implement compute method.
    
    Examples
    --------
        >>> class AnAlgorithm(sfa.Algorithm):
                ...
                ...        
        
        >>> _alg = AnAlgorithm()
        >>> _alg.params = params_obj # Parameters of the algorithm
        >>> _alg.data = data_obj # Data to be analyzed by the algorithm
        >>> _alg.initialize()
        >>> res = _alg.compute()
        
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

    def initialize(self, network=True, ba=True):
        if network:
            self._initialize_network()

        if ba:
            self._initialize_basal_activity()


    def _initialize_network(self):
        pass

    def _initialize_basal_activity(self):
        pass

    @abc.abstractmethod
    def compute(self, b):
        raise NotImplementedError("compute() should be implemented")

    @abc.abstractmethod
    def compute_batch(self):
        raise NotImplementedError("compute_batch() should be implemented")

# end of class Algorithm        


class Data(ContainerItem):
    def __init__(self):
        super().__init__()
        self._A = None
        self._n2i = None
        self._i2n = None
        self._dg = None
        self._inputs= None
        self._df_conds = None
        self._df_exp = None
        self._df_ptb = None
        self._names_ptb = None
        self._iadj_to_idf = None
        self._has_link_perturb = None

    def initialize(self,
                   fpath,
                   fname_network="network.sif",
                   fname_ptb="ptb.tsv",
                   fname_conds="conds.tsv",
                   fname_exp="exp.tsv",
                   inputs={}):

        dpath = os.path.dirname(fpath)
        fpath_network = os.path.join(dpath, fname_network)
        fpath_ptb = os.path.join(dpath, fname_ptb)

        A, n2i, dg = sfa.read_sif(fpath_network, as_nx=True)
        self._A = A
        self._n2i = n2i
        self._dg = dg
        self._df_conds = pd.read_table(os.path.join(dpath, fname_conds),
                                       header=0, index_col=0)
        self._df_exp = pd.read_table(os.path.join(dpath, fname_exp),
                                     header=0, index_col=0)

        self._inputs = inputs
        self._df_ptb = pd.read_table(fpath_ptb, index_col=0)
        if any(self._df_ptb.Type == 'link'):
            self._has_link_perturb = True
        else:
            self._has_link_perturb = False

        self._names_ptb = []
        for i, row in enumerate(self._df_conds.iterrows()):
            row = row[1]
            list_name = []  # Target names
            for target in self._df_conds.columns[row.nonzero()]:
                list_name.append(target)
            # end of for
            self._names_ptb.append(list_name)
        # end of for

        # For mapping from the indices of adj. matrix to those of DataFrame
        # (arrange the indices of adj. matrix according to df_exp.columns)
        self._iadj_to_idf = [n2i[x] for x in self._df_exp.columns]

        self._i2n = {idx: name for name, idx in n2i.items()}

    # end of def

    # Read-only members
    @property
    def A(self):  # Adjacency matrix (numpy.ndarray)
        return self._A

    @property
    def n2i(self):  # Name to index mapping (hashable)
        return self._n2i

    @property
    def i2n(self):  # Index to name mapping (hashable)
        return self._i2n

    @property
    def dg(self):  # Directed graph object of NetworkX
        return self._dg

    @property  # List of perturbation targets
    def names_ptb(self):
        return self._names_ptb

    @property  # List of values for perturbation
    def vals_ptb(self):
        return self._vals_ptb

#    @property  # List of perturbation types
#    def types_ptb(self):
#        return self._types_ptb

    @property
    def iadj_to_idf(self):
        return self._iadj_to_idf

    @iadj_to_idf.setter
    def iadj_to_idf(self, arr):
        self._iadj_to_idf = arr

    # Replaceable (assignable) members
    @property
    def inputs(self):  # Input conditions
        return self._inputs

    @inputs.setter
    def inputs(self, obj_dict):
        self._inputs = obj_dict

    @property  # DataFrame of experimental conditions
    def df_conds(self):
        return self._df_conds

    @df_conds.setter
    def df_conds(self, df):
        self._df_conds = df

    @property  # DataFrame of experimental results
    def df_exp(self):
        return self._df_exp

    @df_exp.setter
    def df_exp(self, df):
        self._df_exp = df

    @property  # DataFrame of perturbation information
    def df_ptb(self):
        return self._df_ptb

    @df_ptb.setter
    def df_ptb(self, df):
        self._df_ptb = df

    @property
    def has_link_perturb(self):
        return self._has_link_perturb

    @has_link_perturb.setter
    def has_link_perturb(self, val):
        if not isinstance(val, bool):
            raise TypeError("has_link_perturb should be boolean.")
        self._has_link_perturb = val

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