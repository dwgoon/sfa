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
    ds.create("BORISOV_2009")
    mult_data = ds["BORISOV_2009"]  # Multiple data

    # Normalized CPS
    algs["NCPS"] = copy.deepcopy(algs["CPS"])
    algs["NCPS"].params.apply_weight_norm = True

    algs["NAPS"] = copy.deepcopy(algs["PW"])
    algs["NAPS"].params.initialize()
    algs["NAPS"].params.apply_weight_norm = True

    dfs = []
    for alg_abbr, alg in algs.items():

        alg.params.use_rel_change = True
        alg.data = sfa.get_avalue(mult_data)

        # Initialize the network and matrices only once
        alg.initialize(data=False)

        results = {}
        for abbr, data in mult_data.items():
            alg.data = data

            # Do not perform initializing network and matrices multiple times
            alg.initialize(network=False)

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
    df = df[["PW", "NAPS", "CPS", "NCPS", "GS", "NGS", "SP"]]
    df.rename(columns={'PW': 'APS',}, inplace=True)
    df_sort = df.sort_index()
    df_sort.to_csv("algs_borisov_2009.tsv", sep="\t")
# end of main
