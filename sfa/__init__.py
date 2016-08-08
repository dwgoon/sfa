


#from .base import Algorithm
from .containers import AlgorithmSet
from .containers import DataSet

from .utils import *

# The following modules are not determined yet
# from .base import Data
# from .base import Result
# from .manager import JobManager



__all__ = []

__all_containers = ["AlgorithmSet",
                    "DataSet"]

__all_utils = ["FrozenClass",
               "Singleton",
               "read_sif",
               "calc_accuracy"]


__all__ += __all_containers
__all__ += __all_utils