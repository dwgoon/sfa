# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:43:42 2016

@author: dwlee
"""

import sfa.base

def create_algorithm():
    return SignalPropagation()
# end of def
    

class SignalPropagation(sfa.base.Algorithm):
    def __init__(self):
        super().__init__()        
        self._id = "SP"
        self._name = "Signal propagation algorithm"        
    
    def compute(self):
        print("Computing signal propagation ...")