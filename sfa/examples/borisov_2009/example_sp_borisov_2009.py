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
    algs.create("SP")
    alg = algs["SP"]

    ds.create("BORISOV_2009")

    """
    It needs to assign one of the data
    for initializing the common network structure only once.
    """
    alg.params.use_rel_change = True
    alg.params.apply_weight_norm = True
    
    alg.data = sfa.get_avalue(ds["BORISOV_2009"])
    alg.initialize()

    results = {}
    for abbr, data in ds["BORISOV_2009"].items():
        alg.data = data

        # Do not perform initializing network and matrices multiple times
        # alg.initialize()

        alg.compute_batch()
        acc = calc_accuracy(alg.result.df_sim,
                            data.df_exp)

        results[abbr] = acc
    # end of for
        

    df = pd.DataFrame.from_dict(results, orient='index')
    df.columns = ['SP']
    
    df_sort = df.sort_values(by='SP')
    print (df_sort)
    #print(df)
    #df.to_csv("sp_borisov_2009.tsv", sep="\t")
    