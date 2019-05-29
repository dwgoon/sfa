
import os
import codecs
from collections import defaultdict

import numpy as np
import networkx as nx

from .base import Data

__all__ = [
    "read_inputs",
    "read_sif",
    "create_from_sif",
]


def read_inputs(fpath):
    inputs = {}
    with codecs.open(fpath, "r", encoding="utf-8-sig") as fin:
        for line in fin:
            if line.isspace():
                continue

            items = line.split()
            node = items[0].strip()
            defval = float(items[1].strip())
            inputs[node] = defval
    return inputs


def read_sif(fpath, signs={'+':1, '-':-1}, sort=True, as_nx=False):
    dict_links = defaultdict(list)
    set_nodes = set()
    n2i = {}
    with codecs.open(fpath, "r", encoding="utf-8-sig") as fin:
        for line in fin:
            if line.isspace():
                continue

            items = line.strip().split()
            src = items[0]
            trg = items[2]
            sign = items[1]

            set_nodes.add(src)
            set_nodes.add(trg)
            int_sign = signs[sign]
            dict_links[src].append((trg, int_sign))
        # end of for
    # end of with

    if sort == True:
        list_nodes = sorted(set_nodes)
    else:
        list_nodes = list(set_nodes)

    N = len(set_nodes)
    adj = np.zeros((N, N), dtype=np.int)

    for isrc, name in enumerate(list_nodes):
        n2i[name] = isrc  # index of source
    # end of for
    for name_src in n2i:
        isrc = n2i[name_src]
        for name_trg, int_sign in dict_links[name_src]:
            itrg = n2i[name_trg]
            adj[itrg, isrc] = int_sign
        # end of for
    # end of for

    if not as_nx:
        return adj, n2i
    else:  # NetworkX DiGraph
        dg = nx.DiGraph()
        # Add nodes
        for name in list_nodes:
            dg.add_node(name)

        # Add edges (links)
        for name_src in list_nodes:
            for name_trg, sign in dict_links[name_src]:
                dg.add_edge(name_src, name_trg)
                dg.edges[name_src, name_trg]['SIGN'] = sign
                # end of for
        # end of for
        return adj, n2i, dg
        # end of else


# end of def


def create_from_sif(fpath, abbr=None, inputs=None, outputs=None):
    """Create sfv.base.Data object from SIF file.

    Parameters
    ----------
    fpath : str
        Absolute path of SIF file
    abbr : str
        Abbreviation to denote this data object for the network.
    inputs : dict, optional
        Input information with default values
    outputs : sequence, optional
        Output information.

    Returns
    -------
    obj : sfv.base.Data
        Data object with the information of network topology.

    """
    class __Data(Data):
        def __init__(self):
            if abbr:
                self._abbr = abbr
            else:
                self._abbr = os.path.basename(fpath)

            self._name = self._abbr
            A, n2i, dg = read_sif(fpath, as_nx=True)
            self._A = A
            self._n2i = n2i
            self._i2n = {idx: name for name, idx in n2i.items()}
            self._dg = dg
            self._inputs = inputs

            if outputs:
                self._outputs = outputs

            # The following members are not defined due to the lack of data.
            self._df_conds = None
            self._df_exp = None
            self._df_ptb = None
            self._has_link_perturb = False
            self._names_ptb = None
            self._iadj_to_idf = None
            # end of def __init__
    # end of def class

    fname, ext = os.path.splitext(os.path.basename(fpath))
    fname = ''.join([c for c in fname.title() if c.isalnum()])
    fname += "Data"
    __Data.__name__ = fname
    return __Data()