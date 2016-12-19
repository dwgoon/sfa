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

    # Normalized PS
    algs["NAPS"] = copy.deepcopy(algs["APS"])
    algs["NAPS"].abbr = "NAPS"
    algs["NAPS"].params.apply_weight_norm = True

    algs["NCPS"] = copy.deepcopy(algs["CPS"])
    algs["NCPS"].abbr = "NCPS"
    algs["NCPS"].params.apply_weight_norm = True


    ds.create("NELANDER_2008")
    data = ds["NELANDER_2008"]

    results = {}
    for abbr, alg in algs.items():
        alg.data = data
        alg.params.use_rel_change = True
        alg.initialize()
        alg.compute_batch()
        acc = calc_accuracy(alg.result.df_sim,
                            data.df_exp)

        results[abbr] = acc
    # end of for
        

    df = pd.DataFrame.from_dict(results, orient='index')
    df.columns = ['NELANDER_2008']
    df = df.ix[["APS", "NAPS", "CPS", "NCPS", "SS", "NSS", "SP"], :]
    print(df)

    #df_sort = df.sort_values(by='NELANDER_2008')
    #print(df_sort)


# end of main
