# -*- coding: utf-8 -*-

import sfa
from .base import BaseRandomBatchSimulator


class RandomWeightBatchSimulator(BaseRandomBatchSimulator):
    def __init__(self, *args, bounds=(-3,0), **kwargs):
        super().__init__(*args, **kwargs)
        self._lb = bounds[0]
        self._ub = bounds[1]

    def _randomize(self):
        sfa.rand_weights(self._W, self._lb, self._ub, inplace=True)
# end of class

