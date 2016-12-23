# -*- coding: utf-8 -*-

import time
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
    data_abbr = "PEZZE_2012"
    class_type = 'UP'
    algs.create()
    ds.create(data_abbr)
    mult_data = ds[data_abbr]  # Multiple data

    algs.create(["APS", "SS", "NSS", "SP"])

    algs["NAPS"] = copy.deepcopy(algs["APS"])
    algs["NAPS"].abbr = "NAPS"
    algs["NAPS"].params.apply_weight_norm = True
    
    algs["NSP"] = copy.deepcopy(algs["SP"])
    algs["NSP"].abbr = "NSP"
    algs["NSP"].params.apply_weight_norm = True

    dfs_acc = []
    dfs_auroc = []
    dfs_auprc = []
    dfs_time = []
    for alg_abbr, alg in algs.items():

        alg.params.use_rel_change = True
    
        # Initialize the network and matrices only once
        alg.data = sfa.get_avalue(mult_data)
        alg.initialize()

        res_acc = {}
        res_auroc = {}
        res_auprc = {}
        res_time = {}
        for data_abbr, data in mult_data.items():
            alg.data = data
            t_beg = time.time()
            alg.compute_batch()
            t_end = time.time()
            acc = sfa.calc_accuracy(data.df_exp, alg.result.df_sim)
            auroc = sfa.calc_auroc(data.df_exp, alg.result.df_sim, class_type)
            auprc = sfa.calc_auprc(data.df_exp, alg.result.df_sim, class_type)
            res_acc[data_abbr] = acc
            res_auroc[data_abbr] = auroc['mean']
            res_auprc[data_abbr] = auprc['mean']
            res_time[data_abbr] = t_end - t_beg
        # end of for

        df_acc = pd.DataFrame.from_dict(res_acc, orient='index')
        df_acc.columns = [alg_abbr]
        dfs_acc.append(df_acc)
        
        df_auroc = pd.DataFrame.from_dict(res_auroc, orient='index')
        df_auroc.columns = [alg_abbr]
        dfs_auroc.append(df_auroc)       
        
        df_auprc = pd.DataFrame.from_dict(res_auprc, orient='index')
        df_auprc.columns = [alg_abbr]
        dfs_auprc.append(df_auprc)
        
        df_time = pd.DataFrame.from_dict(res_time, orient='index')
        df_time.columns = [alg_abbr]
        dfs_time.append(df_time)

        print ("The computation of %s has been finished..."%(alg_abbr))
    # end of for

    
    def to_tsv(dfs, measure, data_name, alg_names=None):
        measure = measure.lower()
        data_name = data_name.lower()
        df = pd.concat(dfs, axis=1)
        if alg_names:            
            df = df[alg_names]
            
        df_sort = df.sort_index()
        df_sort.to_csv("algs_%s_%s.tsv"%(measure, data_name), sep="\t")
    
    alg_names = ["APS", "SS", "SP", "NAPS", "NSS", "NSP"]
    to_tsv(dfs_acc, 'acc', data_abbr, alg_names)
    to_tsv(dfs_auroc, 'auroc', data_abbr, alg_names)
    to_tsv(dfs_auprc, 'auprc', data_abbr, alg_names)
    to_tsv(dfs_time, 'time', data_abbr, alg_names)
    
#    df_auroc = pd.concat(dfs_auroc, axis=1)
#    df_auroc = df_auroc[["APS", "SS", "SP", "NAPS", "NSS", "NSP"]]
#    df_sort = df_auroc.sort_index()
#    df_sort.to_csv("algs_%s_auroc_%s.tsv"%(data_abbr.lower(),
#                                            roc_type.lower()), sep="\t")
#    
#    df_auprc = pd.concat(dfs_auprc, axis=1)
#    df_auprc = df_auprc[["APS", "SS", "SP", "NAPS", "NSS", "NSP"]]
#    df_sort = df_auprc.sort_index()
#    df_sort.to_csv("algs_%s_auprc_%s.tsv"%(data_abbr.lower(),
#                                            roc_type.lower()), sep="\t")
#    
#    df_auprc = pd.concat(dfs_auprc, axis=1)
#    df_auprc = df_auprc[["APS", "SS", "SP", "NAPS", "NSS", "NSP"]]
#    df_sort = df_auprc.sort_index()
#    df_sort.to_csv("algs_%s_auprc_%s.tsv"%(data_abbr.lower(),
#                                            roc_type.lower()), sep="\t")
# end of main
