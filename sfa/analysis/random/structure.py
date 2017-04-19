# -*- coding: utf-8 -*-

import numpy as np
import sfa
from .base import BaseRandomSimulator


class RandomStructureSimulator(BaseRandomSimulator):
    def __init__(self, *args, nswap=10, nflip=10, noself=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._nswap = nswap
        self._nflip = nflip
        self._noself = noself

    def _randomize(self):
        B = sfa.randflip(self._A, self._nflip)
        B = sfa.randswap(B, self._nswap, self._noself)
        ir, ic = B.nonzero()
        self._W[ir, ic] = B[ir, ic]
# end of class
