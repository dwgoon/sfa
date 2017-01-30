# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import sfa
from sfa import AlgorithmSet
from sfa import DataSet


if __name__ == "__main__":
    np.seterr(all='raise')
    # Create containers for algorithm and data.
    algs = AlgorithmSet()
    ds = DataSet()

    # Load an algorithm and a data.
    algs.create("APS")
    alg = algs["APS"]

    ds.create("SCHLIEMANN_2011")

    """
    It needs to assign one of the data
    for initializing the common network structure only once.
    """
    alg.params.use_rel_change = False
    alg.data = sfa.get_avalue(ds["SCHLIEMANN_2011"])

    # Initialize the network and matrices only once
    alg.initialize(data=False)

    results = {}
    for abbr, data in ds["SCHLIEMANN_2011"].items():
        alg.data = data

        # Do not perform initializing network and matrices multiple times
        alg.initialize(network=False)

        alg.compute_batch()
        acc = sfa.calc_accuracy(alg.result.df_sim,
                                data.df_exp)

        results[abbr] = acc
    # end of for

    df = pd.DataFrame.from_dict(results, orient='index')
    df.columns = ['APS']

    df_sort = df.sort_values(by='APS')
    print(df_sort)
