# -*- coding: utf-8 -*-
import copy
import pandas as pd

import sfa
from sfa import calc_accuracy
from sfa import AlgorithmSet
from sfa import DataSet


if __name__ == "__main__":

    # Create containers for algorithm and data.
    algs = AlgorithmSet()
    ds = DataSet()

    # Load an algorithm and a data.
    algs.create()
    ds.create("SCHLIEMANN_2011")
    mult_data = ds["SCHLIEMANN_2011"]  # Multiple data

    # Normalized PS
    algs["NAPS"] = copy.deepcopy(algs["APS"])
    algs["NAPS"].abbr = "NAPS"
    algs["NAPS"].params.apply_weight_norm = True

    algs["NCPS"] = copy.deepcopy(algs["CPS"])
    algs["NCPS"].abbr = "NCPS"
    algs["NCPS"].params.apply_weight_norm = True


    dfs = []
    for alg_abbr, alg in algs.items():

        alg.params.use_rel_change = True
        alg.data = sfa.get_avalue(mult_data)

        # Initialize the network and matrices only once
        alg.initialize()

        results = {}
        for abbr, data in mult_data.items():
            alg.data = data
            alg.compute_batch()
            acc = calc_accuracy(alg.result.df_sim,
                                data.df_exp)

            results[abbr] = acc
        # end of for

        df = pd.DataFrame.from_dict(results, orient='index')
        df.columns = [alg_abbr]
        dfs.append(df)
        print ("The computation of %s has been finished..."%(alg_abbr))
    # end of for

    df = pd.concat(dfs, axis=1)
    df = df[["APS", "NAPS", "CPS", "NCPS", "SS", "NSS", "SP"]]
    df_sort = df.sort_index()
    df_sort.to_csv("algs_schliemann_2011.tsv", sep="\t")
# end of main
