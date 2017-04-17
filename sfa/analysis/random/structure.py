# -*- coding: utf-8 -*-

import numpy as np
import sfa
from .base import BaseRandomSimulator


class RandomStructureSimulator(BaseRandomSimulator):
    def __init__(self, *args, nswap=10, noself=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._nswap = nswap
        self._noself = noself

    def _randomize(self):
        B = sfa.randswap(self._A, self._nswap, self._noself)
        ir, ic = B.nonzero()
        self._W[ir, ic] = B[ir, ic]
# end of class
