
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
            items = line.split()
            node = items[0].strip()
            defval = float(items[1].strip())
            inputs[node] = defval
    return inputs


def read_sif(fpath, str_act='+', str_inh='-', sort=True, as_nx=False):
    dict_links = defaultdict(list)
    set_nodes = set()
    name_to_idx = {}
    with codecs.open(fpath, "r", encoding="utf-8-sig") as fin:
        for line in fin:
            items = line.strip().split()
            src = items[0]
            tgt = items[2]
            sign = items[1]

            set_nodes.add(src)
            set_nodes.add(tgt)
            if sign == str_act:
                dict_links[src].append((tgt, 1))
            elif sign == str_inh:
                dict_links[src].append((tgt, -1))
            else:
                raise ValueError("Undefined link type: %s"%(sign))

        # end of for
    # end of with

    if sort == True:
        list_nodes = sorted(set_nodes)
    else:
        list_nodes = list(set_nodes)

    N = len(set_nodes)
    adj = np.zeros((N, N), dtype=np.int)

    for isrc, name in enumerate(list_nodes):
        name_to_idx[name] = isrc  # index of source
    # end of for
    for name_src in name_to_idx:
        isrc = name_to_idx[name_src]
        for name_tgt, sign in dict_links[name_src]:
            itgt = name_to_idx[name_tgt]
            adj[itgt, isrc] = sign
            # end of for
    # end of for

    if not as_nx:
        return adj, name_to_idx
    else:  # NetworkX DiGraph
        dg = nx.DiGraph()
        # Add nodes
        for name in list_nodes:
            dg.add_node(name)

        # Add edges (links)
        for name_src in list_nodes:
            for name_tgt, sign in dict_links[name_src]:
                dg.add_edge(name_src, name_tgt)
                dg.edges[name_src, name_tgt]['SIGN'] = sign
                # end of for
        # end of for
        return adj, name_to_idx, dg
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