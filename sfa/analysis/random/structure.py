# -*- coding: utf-8 -*-

import sfa
from .base import BaseRandomBatchSimulator


class RandomStructureBatchSimulator(BaseRandomBatchSimulator):
    def __init__(self, *args, nswap=10, nflip=10, noself=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._nswap = nswap
        self._nflip = nflip
        self._noself = noself

    def _randomize(self):
        B = sfa.rand_flip(self._A, self._nflip)
        B = sfa.rand_swap(B, self._nswap, self._noself)
        ir, ic = B.to_numpy().nonzero()
        self._W[ir, ic] = B[ir, ic]
# end of class
