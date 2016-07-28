# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 15:05:25 2016

@author: dwlee
"""

from abc import ABCMeta, abstractmethod

from .base import Algorithm
from .base import Data

class JobManager(object):
    __metaclass__ = ABCMeta

    def add_algorithm(self, alg):
        
        if not isinstance(alg, Algorithm):
            raise TypeError("The algorithm should be " \
                            "a subclass of sfa.base.Algorithm.")
    # end of def add_algorithm
                            
    def add_data(self, data):

        if not isinstance(data, Data):
            raise TypeError("The data should be " \
                            "a subclass of sfa.base.Data.")
                            
    # end of def add_data                            

    #def (self, nprocs=False):
    #    pass
    
    # end of def process
    
    def _create_job(self, alg, data):
        pass
    
    def _process_parallel(self):
        pass
    
    def _process_single(self):
        pass
        
        
        