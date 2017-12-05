# -*- coding: utf-8 -*-
import sys
if sys.version_info <= (2, 8):
    from builtins import super

import numpy as np
import networkx as nx
from networkx import shortest_paths as nxsp

__all__ = ["max_shortest_path_length",]


def max_shortest_path_length(nxdg):
    """Find the maximum length of the shortest path
    """
    all_spl = nxsp.all_pairs_shortest_path_length(nxdg)
    max_spl = 0
    for src, targets in all_spl:
        for tgt, spl in targets.items():
            if spl > max_spl:
                max_spl = spl
        # end of for
    # end of for
    return max_spl