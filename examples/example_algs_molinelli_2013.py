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

    ds.create("MOLINELLI_2013")
    data = ds["MOLINELLI_2013"]

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
    df.columns = ['MOLINELLI_2013']
    
    df_sort = df.sort_values(by='MOLINELLI_2013')
    print(df_sort)


# end of main
