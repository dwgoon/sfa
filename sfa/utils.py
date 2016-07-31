# -*- coding: utf-8 -*-

import codecs
from collections import defaultdict

import numpy as np
import networkx as nx


def read_sif(filename, sym_pos='+', sort=True, as_nx=False):
    dict_links = defaultdict(list)
    set_nodes = set()
    name_to_idx = {}
    with codecs.open(filename, "r", encoding="utf-8") as f_in:
        for line in f_in:
            items =  line.strip().split()
            src = items[0]
            tgt = items[2]
            sign = items[1]

            set_nodes.add( src )
            set_nodes.add( tgt )
            if sign == sym_pos:
                dict_links[src].append( (tgt, 1) )
            else:
                dict_links[src].append( (tgt, -1) )
        # end of for
    # end of with

    if sort == True:
        list_nodes = sorted(set_nodes)
    else:
        list_nodes = list(set_nodes)
            
    if not as_nx:
        N = len(set_nodes)
        adj = np.zeros((N,N), dtype=np.int)
        
        for isrc, name in enumerate(list_nodes):
            name_to_idx[name] = isrc # index of source
        # end of for
        for name_src in name_to_idx:
            isrc = name_to_idx[name_src]
            for name_tgt, sign in dict_links[name_src]:
                itgt = name_to_idx[name_tgt]
                adj[itgt, isrc] = sign    
            # end of for
        # end of for
        return adj, name_to_idx
    else: # NetworkX DiGraph
        dg = nx.DiGraph()
        # Add nodes
        for name in list_nodes:
            dg.add_node(name)
            
        # Add edges (links)
        for name_src in list_nodes:
            for name_tgt, sign in dict_links[name_src]:
                dg.add_edge(name_src, name_tgt,
                            attr_dict={'sign': sign})
            # end of for
        # end of for
        return dg
    # end of else
# end of def

"""
def convert_networkx_digraph(adj, name_to_idx=None):
    from sys import modules
    try:
        nx = modules["networkx"]
    except:
        nx = __import__("networkx")

    dg = nx.DiGraph()

    if name_to_idx is not None:
        idx_to_name = len(name_to_idx)*['']
        for key, val in name_to_idx.items():
            idx_to_name[val] = key

        for i, row in enumerate(adj):
            tgt = idx_to_name[i]
            for j, sign in enumerate(row):
                if sign == 0:
                    continue
                # end of if
                src = idx_to_name[j]
                if sign>0:
                    dg.add_edge(src, tgt, sign=+1)
                else:
                    dg.add_edge(src, tgt, sign=-1)
            # end of for
        # end of for
    else:
        for i, row in enumerate(adj):
            for j, sign in enumerate(row):
                if sign == 0:
                    continue
                # end of if
                if sign>0:
                    dg.add_edge(j, i, sign=+1)
                else:
                    dg.add_edge(j, i, sign=-1)
            # end of for
        # end of for
    # end of else
    return dg
"""