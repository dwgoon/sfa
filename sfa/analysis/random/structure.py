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
        # Replace _W in place so stale nonzero entries from prior
        # iterations are cleared (positions that became zero in B).
        self._W[:] = B
# end of class
