# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:03:58 2016

@author: dwlee
"""

import sfa.base

def create_algorithm(abbr):
    return PathwayWiring(abbr)
# end of def
    

class PathwayWiring(sfa.base.Algorithm):
    def __init__(self, abbr):
        super().__init__(abbr)        
        self._name = "Feiglin's pathway wiring algorithm"       

    
    def compute(self):
        print("Computing pathway wiring ...")