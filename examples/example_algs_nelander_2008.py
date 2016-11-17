# -*- coding: utf-8 -*-

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

    ds.create("NELANDER_2008")
    data = ds["NELANDER_2008"]

    results = {}
    for abbr, alg in algs.items():
        alg.data = data
        alg.params.use_rel_change = True
        alg.initialize()

        # Do not perform initializing network and matrices multiple times
        alg.initialize(network=False)

        alg.compute_batch()
        acc = calc_accuracy(alg.result.df_sim,
                            data.df_exp)

        results[abbr] = acc
    # end of for
        

    df = pd.DataFrame.from_dict(results, orient='index')
    df.columns = ['NELANDER_2008']
    
    df_sort = df.sort_values(by='NELANDER_2008')
    print(df_sort)


# end of main
