# -*- coding: utf-8 -*-
import copy
import pandas as pd

import sfa
from sfa import AlgorithmSet
from sfa import DataSet


if __name__ == "__main__":

    # Create containers for algorithm and data.
    algs = AlgorithmSet()
    ds = DataSet()

    # Load an algorithm and a data.
    algs.create()
    ds.create("BORISOV_2009")
    mult_data = ds["BORISOV_2009"]  # Multiple data

    algs.create(["APS", "SS", "NSS", "SP"])


    algs["NAPS"] = copy.deepcopy(algs["APS"])
    algs["NAPS"].abbr = "NAPS"
    algs["NAPS"].params.apply_weight_norm = True

    
    algs["NSP"] = copy.deepcopy(algs["SP"])
    algs["NSP"].abbr = "NSP"
    algs["NSP"].params.apply_weight_norm = True

    dfs_acc = []
    dfs_rocauc = []
    for alg_abbr, alg in algs.items():

        alg.params.use_rel_change = True
    
        # Initialize the network and matrices only once
        alg.data = sfa.get_avalue(mult_data)
        alg.initialize()

        res_acc = {}
        res_rocauc = {}
        for abbr, data in mult_data.items():
            alg.data = data
            alg.compute_batch()
            acc = sfa.calc_accuracy(data.df_exp, alg.result.df_sim)
            _, _, rocauc = sfa.calc_roc_auc(data.df_exp, alg.result.df_sim)
            res_acc[abbr] = acc
            res_rocauc[abbr] = rocauc['mean']
        # end of for

        df_acc = pd.DataFrame.from_dict(res_acc, orient='index')
        df_acc.columns = [alg_abbr]
        dfs_acc.append(df_acc)
        
        df_rocauc = pd.DataFrame.from_dict(res_rocauc, orient='index')
        df_rocauc.columns = [alg_abbr]
        dfs_rocauc.append(df_rocauc)       

        print ("The computation of %s has been finished..."%(alg_abbr))
    # end of for

    df_acc = pd.concat(dfs_acc, axis=1)
    df_acc = df_acc[["APS", "SS", "SP", "NAPS", "NSS", "NSP"]]
    df_sort = df_acc.sort_index()
    df_sort.to_csv("algs_borisov_2009_acc.tsv", sep="\t")
    
    
    df_rocauc = pd.concat(dfs_rocauc, axis=1)
    df_rocauc = df_rocauc[["APS", "SS", "SP", "NAPS", "NSS", "NSP"]]
    df_sort = df_rocauc.sort_index()
    df_sort.to_csv("algs_borisov_2009_rocauc.tsv", sep="\t")
# end of main
