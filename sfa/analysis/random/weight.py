# -*- coding: utf-8 -*-

import numpy as np
from .base import BaseRandomSimulator


class RandomWeightSimulator(BaseRandomSimulator):
    def __init__(self, *args, bounds=(-3,0), **kwargs):
        super().__init__(*args, **kwargs)
        self._lb = bounds[0]
        self._ub = bounds[1]

    def _randomize(self):
        self._randomize_weights()
                
    def _randomize_weights(self):
        weights_rand = 10**np.random.uniform(self._lb, self._ub,
                                             size=(self._num_links,))
        for i in range(self._num_links):
            p, q = self._ir[i], self._ic[i]
            self._W[p, q] = weights_rand[i] * self._S[p, q]
        # end of for
# end of class

