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

    data = ds.create("MOLINELLI_2013")

    # Load an algorithm and a data.
    algs.create(["APS", "SS", "NSS", "SP"])

    algs["NAPS"] = copy.deepcopy(algs["APS"])
    algs["NAPS"].abbr = "NAPS"
    algs["NAPS"].params.apply_weight_norm = True

    algs["NSP"] = copy.deepcopy(algs["SP"])
    algs["NSP"].abbr = "NSP"
    algs["NSP"].params.apply_weight_norm = True


    results = {}
    for abbr, alg in algs.items():        
        alg.params.use_rel_change = True
        alg.data = data
        alg.initialize()

        alg.compute_batch()
        df_sim = alg.result.df_sim
        acc, cons = calc_accuracy(df_sim,
                                  data.df_exp,
                                  get_cons=True)

        results[abbr] = acc
        print ("%s: %.3f"%(abbr, acc))
    # end of for
        
    df = pd.DataFrame.from_dict(results, orient='index')
    df.columns = ['MOLINELLI_2013']
    #df = df[["APS", "SS", "SP", "NAPS", "NSS", "NSP"]]
    print(df.ix[["APS", "SS", "SP", "NAPS", "NSS", "NSP"], 0])

    #df_sort = df.sort_values(by='MOLINELLI_2013')
    #print(df_sort)


# end of main
