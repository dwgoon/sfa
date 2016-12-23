# -*- coding: utf-8 -*-
import copy
import pandas as pd

import numpy as np

import sfa
from sfa import calc_accuracy
from sfa import AlgorithmSet
from sfa import DataSet


if __name__ == "__main__":

    # Create containers for algorithm and data.
    algs = AlgorithmSet()
    ds = DataSet()

    # Load an algorithm and a data.
    data_abbr = "NELANDER_2008"
    roc_type = 'DN'
    data = ds.create(data_abbr)

    algs.create(["APS", "SS", "NSS", "SP"])

    algs["NAPS"] = copy.deepcopy(algs["APS"])
    algs["NAPS"].abbr = "NAPS"
    algs["NAPS"].params.apply_weight_norm = True

    algs["NSP"] = copy.deepcopy(algs["SP"])
    algs["NSP"].abbr = "NSP"
    algs["NSP"].params.apply_weight_norm = True


    alpha = 0.5
    dfs_acc = []
    dfs_rocauc = []
    vals_EGF = np.logspace(-2, 2, 5)
    #vals_EGF = [1.0]
    for alg_abbr, alg in algs.items():

        alg.params.use_rel_change = True
        alg.params.alpha = alpha

        res_acc = {}
        res_rocauc = {}
        for val in vals_EGF:
            data.inputs['EGF'] = val
            alg.data = data
            alg.initialize()
            alg.compute_batch()
            acc = sfa.calc_accuracy(data.df_exp, alg.result.df_sim)
            rocauc = sfa.calc_roc(data.df_exp, alg.result.df_sim, roc_type)
            res_acc['EGF_%.1e'%val] = acc
            res_rocauc['EGF_%f'%val] = rocauc['mean']
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
    df_sort.to_csv("algs_%s_acc.tsv" % (data_abbr.lower()), sep="\t")

    df_rocauc = pd.concat(dfs_rocauc, axis=1)
    df_rocauc = df_rocauc[["APS", "SS", "SP", "NAPS", "NSS", "NSP"]]
    df_sort = df_rocauc.sort_index()
    df_sort.to_csv("algs_%s_rocauc_%s.tsv" % (data_abbr.lower(),
                                              roc_type.lower()), sep="\t")
# end of main
