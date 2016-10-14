# -*- coding: utf-8 -*-

import pandas as pd

from sfa import calc_accuracy
from sfa import AlgorithmSet
from sfa import DataSet


if __name__ == "__main__":

    # Create containers for algorithm and data.
    algs = AlgorithmSet()
    ds = DataSet()

    # Load an algorithm and a data.
    algs.create("SP")
    alg_sp = algs["SP"]

    ds.create("BORISOV_2009")

    """
    It needs to assign one of the data
    for initializing the common network structure only once.
    """
    alg_sp.params.use_rel_change = True
    alg_sp.data = ds["BORISOV_2009"]["AUC_EGF=0.1+I=100"]

    # Initialize the network and matrices only once
    alg_sp.initialize(data=False)

    results = {}
    for abbr, data in ds["BORISOV_2009"].items():
        alg_sp.data = data

        # Do not perform initializing network and matrices multiple times
        alg_sp.initialize(network=False)

        alg_sp.compute_batch()
        acc = calc_accuracy(alg_sp.result.df_sim,
                            data.df_exp)

        results[abbr] = acc
    # end of for
        

    df = pd.DataFrame.from_dict(results, orient='index')
    df.columns = ['SP']
    
    df_sort = df.sort_values(by='SP')
    print(df_sort)
    