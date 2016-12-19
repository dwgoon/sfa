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

    ds.create("MOLINELLI_2013")
    data = ds["MOLINELLI_2013"]

    #data.df_conds.drop(data.df_conds.columns[:17], axis=1, inplace=True)
    #data.df_exp.drop(data.df_exp.columns[17:], axis=1, inplace=True)


    # Load an algorithm and a data.
    algs.create()

    # algs["SPN"] = copy.deepcopy(algs["SP"])
    # algs["SPN"].params.apply_weight_norm = False
    # algs["SPN"].abbr = "SPN"

    # Normalized PS
    algs["NAPS"] = copy.deepcopy(algs["APS"])
    algs["NAPS"].abbr = "NAPS"
    algs["NAPS"].params.apply_weight_norm = True

    algs["NCPS"] = copy.deepcopy(algs["CPS"])
    algs["NCPS"].abbr = "NCPS"
    algs["NCPS"].params.apply_weight_norm = True



    results = {}
    for abbr, alg in algs.items():        
        alg.params.use_rel_change = True
        alg.data = data
        alg.initialize()

        alg.compute_batch()
        #df_sim = alg.result.df_sim.ix[:, :17]
        df_sim = alg.result.df_sim
        acc, cons = calc_accuracy(df_sim,
                                  data.df_exp,
                                  get_cons=True)

        results[abbr] = acc
        print ("%s: %.3f"%(abbr, acc))
    # end of for
        
    df = pd.DataFrame.from_dict(results, orient='index')
    df.columns = ['MOLINELLI_2013']
    df = df.ix[["APS", "NAPS", "CPS", "NCPS", "SS", "NSS", "SP"], :]
    print(df)

    #df_sort = df.sort_values(by='MOLINELLI_2013')
    #print(df_sort)


# end of main
