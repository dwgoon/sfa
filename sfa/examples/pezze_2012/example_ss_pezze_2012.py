# -*- coding: utf-8 -*-

import pandas as pd

import sfa
from sfa import AlgorithmSet
from sfa import DataSet


if __name__ == "__main__":

    # Create containers for algorithm and data.
    algs = AlgorithmSet()
    ds = DataSet()

    # Load an algorithm and a data.
    algs.create("SS")
    alg = algs["SS"]

    ds.create("PEZZE_2012")

    """
    It needs to assign one of the data
    for initializing the common network structure only once.
    """
    alg.params.use_rel_change = True
    #alg.params.apply_weight_norm = True
    alg.data = sfa.get_avalue(ds["PEZZE_2012"])

    # Initialize the network and matrices only once
    alg.initialize(data=False)

    results = {}
    for abbr, data in ds["PEZZE_2012"].items():
        alg.data = data

        # Do not perform initializing network and matrices multiple times
        alg.initialize(network=False)

        alg.compute_batch()
        acc = sfa.calc_accuracy(alg.result.df_sim,
                                data.df_exp)

        results[abbr] = acc
    # end of for

    df = pd.DataFrame.from_dict(results, orient='index')
    df.columns = ['SS']
    
    df_sort = df.sort_values(by='SS')
    print(df_sort)
    