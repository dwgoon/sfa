# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 00:22:58 2016
"""

import numpy as np
import pandas as pd
import networkx as nx

from sysbio import read_sif
from sysbio import get_idx_to_name
from sysbio import convert_networkx_digraph

from sysbio.other.feiglin import feiglin_wiring

def calc_accuracy(df1, df2):
    # Count the same signs between the results of sim. and exp.
    np.sign(df1) + np.sign(df2)
    num_total = df1.count().sum()
    diff_abs = np.abs( np.sign(df1) - np.sign(df2) )
    num_success = (diff_abs == 0).sum(axis=1).sum()

    return (num_success)/np.float(num_total)


def wire_single_path(dg, ba, path):
    F = ba
    for i in range(len(path)-1):
        src = path[i]
        tgt = path[i+1]
        w = dg.edge[src][tgt]['weight']
        F *= w
    # end of for
    return F


def wire_all_paths(dg, ba, src, tgt):
    paths = nx.all_simple_paths(dg, src, tgt)
    E = 0
    # Calculate the F for each path
    for i, path in enumerate(paths):
        F = wire_single_path(dg, ba, path)
        E += F
    # end of for

    # Apply the effect of perturbation on the target itself
    #if ptb == tgt:
    #    F = calc_F(dg, [tgt], w)
    #    E += F
    return E
        
    
def wire(dg, names_ba_se, val_ba_se, n2i):
    """
    dg: NetworkX.DiGraph object including information of signs and weights
    names_ba_se: names of basal activities in a single experiment
    val_ba_se: values of basal activities in a single experiment
    n2i: name to index mapper
    """
    CE = np.zeros((dg.number_of_nodes(),), dtype=np.float)

    for tgt in dg.nodes_iter():
        Et = 0.0
        for i, src in enumerate(names_ba_se):
            ba = val_ba_se[i]        
            #print(name_src, ba)
            Et += wire_all_paths(dg, ba, src, tgt)            
        # end of for
        CE[ n2i[tgt] ] = Et
    # end of for
    return CE
# end of def wire
    
    
if __name__ == "__main__":

    str_cond = "EGF+I"
    tb_conds = pd.read_table("ba.tsv",
                             header=0, index_col=0)
    tb_exp_res = pd.read_table("exp_auc_%s.tsv"%(str_cond),
                               header=0, index_col=0)
    sim_results = np.zeros(tb_exp_res.shape, dtype=np.float)

    A, name_to_idx = read_sif("network.sif")
    
    #A, name_to_idx = read_sif("Borisov2009msb_SHP2_without_AKTRAF.txt")
    n2i = name_to_idx
    i2n = {idx:name for name, idx in n2i.items()}
    
    W = 0.5*A
    dg = convert_networkx_digraph(A, name_to_idx)
    
    # Assign weights for the edges
    for edge in dg.edges():
        src = edge[0]
        tgt = edge[1]
        w = W[n2i[tgt], n2i[src]]
        dg.edge[src][tgt]['weight'] = w
    
    nodes = get_idx_to_name(name_to_idx)
    N = A.shape[0]

    iadj_to_itb = list(map(lambda x: name_to_idx[x], tb_exp_res.columns))

    b = np.zeros(N, dtype=np.float)
    names_ba = []
    vals_ba = []
    for i, row in enumerate(tb_conds.iterrows()):
        row = row[1]
        list_name = []  # Indices
        list_val = []  # Values
        for target in tb_conds.columns[row.nonzero()]:
            list_name.append(target)
            list_val.append(row[target])
        # end of for
        names_ba.append(list_name)
        vals_ba.append(list_val)
    # end of for
        
    # Main loop of the simulation
    #for i, row in enumerate(tb_conds.iterrows()):
    for i, names_ba_se in enumerate(names_ba):
        vals_ba_se = vals_ba[i]        
        arr_CE = wire(dg, names_ba_se, vals_ba_se, n2i,) 
        

        #print np.abs(x0).sum()

        # xs_ratio = np.log2(xs_exp/xs_cont)
        sim_results[i, :] = arr_CE[ iadj_to_itb ]
    # end of for


    tb_sim_res = pd.DataFrame(sim_results,
                              index=tb_exp_res.index,
                              columns=tb_exp_res.columns)

    # Abandon the proteins which are not included in the exp. results.
    tb_sim_res = tb_sim_res[ tb_exp_res.columns ]



    ## Count the same signs between the results of sim. and exp.
    #np.sign(tb_sim_res) + np.sign(tb_exp_res)
    #
    #num_total = tb_sim_res.count().sum()
    #
    ## The value should be 2 if the signs are equal.
    #num_fail = (np.abs( np.sign(tb_sim_res) + np.sign(tb_exp_res) ) != 2).sum(axis=1).sum()
    #
    #accuracy = (num_total - num_fail)/np.float(num_total)



#    print np.abs(np.sign(tb_sim_res) - np.sign(tb_exp_res)) == 0
#    (np.abs(np.sign(tb_sim_res) - np.sign(tb_exp_res)) == 0).T.to_clipboard()
#
#    print np.sign(tb_sim_res)
#    np.sign(tb_sim_res).T.to_clipboard()

    print("accuracy (total): %.3f"%calc_accuracy(tb_sim_res, tb_exp_res))
    print("accuracy (output): %.3f"%calc_accuracy(tb_sim_res[['ERK', 'AKT']],
                                                  tb_exp_res[['ERK', 'AKT']]))
                                               
                                               
