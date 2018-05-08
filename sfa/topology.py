# -*- coding: utf-8 -*-
import sys
if sys.version_info <= (2, 8):
    from builtins import super

import numpy as np
import pandas as pd
import networkx as nx
from networkx import shortest_paths as nxsp

__all__ = ["max_spl",
           "splo"]


def max_spl(nxdg):
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


def splo(nxdg, sources, outputs, rtype='df'):
    """Calculate the shortest path length
       from each source node to the outputs.
       SPLO represents
       'shortest path length to output'.

       Parameters
       ----------
       nxdg: NetworkX.DiGraph
           A directed network in NetworkX.
       sources: list (or iterable) of str
           Names of source nodes in nxdg.
       outputs: list (or iterable) of str
           Names of output nodes in nxdg.
       rtype: str (optional)
           Return object type: 'df' or 'dict'.

       Returns
       -------
       splo: dict
           All the shortest path lengths
           to the outputs.

    """
    if isinstance(outputs, str):
        outputs = [outputs]

    dict_splo = {}
    for trg in outputs:
        dict_splo[trg] = {}
        for src in sources:
            try:
                splo = nxsp.shortest_path_length(nxdg,
                                                 src,
                                                 trg)
            except nx.NetworkXNoPath:
                continue  # splo = np.inf

            dict_splo[trg][src] = splo

    if rtype == 'df':
        df = pd.DataFrame(dict_splo)
        df.index.name = 'Source'
        return df

    return dict_splo
