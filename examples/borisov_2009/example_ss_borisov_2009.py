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
    algs.create("SS")
    alg = algs["SS"]

    ds.create("BORISOV_2009")

    """
    It needs to assign one of the data
    for initializing the common network structure only once.
    """
    alg.params.use_rel_change = True
    alg.data = sfa.get_avalue(ds["BORISOV_2009"])

    # Initialize the network and matrices only once
    alg.initialize(data=False)

    results = {}
    for abbr, data in ds["BORISOV_2009"].items():
        alg.data = data

        # Do not perform initializing network and matrices multiple times
        alg.initialize(network=False)

        alg.compute_batch()
        acc = calc_accuracy(alg.result.df_sim,
                            data.df_exp)

        results[abbr] = acc
    # end of for
        

    df = pd.DataFrame.from_dict(results, orient='index')
    df.columns = ['SS']
    
    #df_sort = df.sort_values(by='SS')
    #print(df_sort)
    print(df)
    df.to_csv("gs_borisov_2009.tsv", sep="\t")

# end of main
