# -*- coding: utf-8 -*-
"""
@author: dwlee
"""
# This module is for the purpose of testing
# (This algorithm is not included in Algorithms object)

import sfa.base


def create_algorithm(abbr):
    return ThisAlgorithm(abbr)


class ThisAlgorithm(sfa.base.Algorithm):

    def compute_panel(self):
        pass
    

    