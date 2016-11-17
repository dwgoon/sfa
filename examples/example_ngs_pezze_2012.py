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
    algs.create("NGS")
    alg_sp = algs["NGS"]

    ds.create("PEZZE_2012")

    """
    It needs to assign one of the data
    for initializing the common network structure only once.
    """
    alg_sp.params.use_rel_change = True
    alg_sp.data = sfa.get_avalue(ds["PEZZE_2012"])

    # Initialize the network and matrices only once
    alg_sp.initialize(data=False)

    results = {}
    for abbr, data in ds["PEZZE_2012"].items():
        alg_sp.data = data

        # Do not perform initializing network and matrices multiple times
        alg_sp.initialize(network=False)

        alg_sp.compute_batch()
        acc = sfa.calc_accuracy(alg_sp.result.df_sim,
                                data.df_exp)

        results[abbr] = acc
    # end of for

    df = pd.DataFrame.from_dict(results, orient='index')
    df.columns = ['NGS']
    
    df_sort = df.sort_values(by='NGS')
    print(df_sort)
    