# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 17:35:52 2016

@author: dwlee
"""

class AlgorithmFactory(object):
    
    @classmethod
    def create(cls, name=None):
        return name