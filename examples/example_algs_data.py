# -*- coding: utf-8 -*-

import time
import copy
import pandas as pd

import sfa
from sfa import AlgorithmSet
from sfa import DataSet


def append_dataframe(dfs, res, alg_abbr):
    df = pd.DataFrame.from_dict(res, orient='index')
    df.columns = [alg_abbr]
    dfs.append(df)
    

def to_tsv(measure, dfs, data_name, alg_names=None, class_type=None):
    measure = measure.lower()
    data_name = data_name.lower()
    df = pd.concat(dfs, axis=1)
    if alg_names:            
        df = df[alg_names]
        
    df_sort = df.sort_index()
    if class_type:
        class_type = class_type.lower()
        fstr = "algs_%s_%s_%s.tsv"%(data_name, measure, class_type)
    else:
        fstr = "algs_%s_%s.tsv"%(data_name, measure)
        
    df_sort.to_csv(fstr, sep="\t")
    
    

if __name__ == "__main__":

    # Create containers for algorithm and data.
    algs = AlgorithmSet()
    ds = DataSet()

    # Load an algorithm and a data.
    data_abbr = "BORISOV_2009"
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
    dfs_auroc_up = []
    dfs_auroc_dn = []
    dfs_auprc_up = []
    dfs_auprc_dn = []
    dfs_time = []
    
    for alg_abbr, alg in algs.items():
        alg.params.use_rel_change = True
    
        # Initialize the network and matrices only once
        alg.data = sfa.get_avalue(mult_data)
        alg.initialize()
        
        res_acc = {}
        res_auroc_up = {}        
        res_auroc_dn = {}
        res_auprc_up = {}
        res_auprc_dn = {}
        res_time = {}
        for cond, data in mult_data.items():  # cond: experimental condition
            alg.data = data
            t_beg = time.time()
            alg.compute_batch()
            t_end = time.time()
            acc = sfa.calc_accuracy(data.df_exp, alg.result.df_sim)
            res_acc[cond] = acc
            
            auroc = sfa.calc_auroc(data.df_exp, alg.result.df_sim, 'UP')
            res_auroc_up[cond] = auroc['mean']
            
            auroc = sfa.calc_auroc(data.df_exp, alg.result.df_sim, 'DN')
            res_auroc_dn[cond] = auroc['mean']
            
            auprc = sfa.calc_auprc(data.df_exp, alg.result.df_sim, 'UP')
            res_auprc_up[cond] = auprc['mean']
            
            auprc = sfa.calc_auprc(data.df_exp, alg.result.df_sim, 'DN')
            res_auprc_dn[cond] = auprc['mean']
            
            res_time[cond] = t_end - t_beg
        # end of for

        append_dataframe(dfs_acc, res_acc, alg_abbr)
        append_dataframe(dfs_auroc_up, res_auroc_up, alg_abbr)
        append_dataframe(dfs_auroc_dn, res_auroc_dn, alg_abbr)
        append_dataframe(dfs_auprc_up, res_auprc_up, alg_abbr)
        append_dataframe(dfs_auprc_dn, res_auprc_dn, alg_abbr)
        append_dataframe(dfs_time, res_time, alg_abbr)

        print ("The computation of %s has been finished..."%(alg_abbr))
    # end of for
    
    alg_names = ["APS", "SS", "SP", "NAPS", "NSS", "NSP"]
    to_tsv('accuracy', dfs_acc, data_abbr, alg_names)
    to_tsv('auroc', dfs_auroc_up, data_abbr, alg_names, 'UP')
    to_tsv('auroc', dfs_auroc_dn, data_abbr, alg_names, 'DN')
    to_tsv('auprc', dfs_auprc_up, data_abbr, alg_names, 'UP')
    to_tsv('auprc', dfs_auprc_dn, data_abbr, alg_names, 'DN')
    to_tsv('time', dfs_time, data_abbr, alg_names)
    
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
