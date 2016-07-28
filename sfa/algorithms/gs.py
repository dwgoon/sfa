# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:03:58 2016

@author: dwlee
"""

import sfa

def create_algorithm():
    return GaussianSmoothing()
# end of def
    

class GaussianSmoothing(sfa.base.Algorithm):
    def __init__(self):
        super().__init__()        
        self._id = "GS"
        self._name = "Gaussian smoothing algorithm"       
    
    def compute(self):
        print("Computing Gaussian smoothing ...")