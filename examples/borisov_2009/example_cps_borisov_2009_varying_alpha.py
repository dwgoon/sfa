# -*- coding: utf-8 -*-

import numpy as np
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
    algs.create("CPS")
    alg = algs["CPS"]

    ds.create("BORISOV_2009")

    """
    It needs to assign one of the data
    for initializing the common network structure only once.
    """
    alg.params.initialize()
    #alg.params.use_rel_change = True
    #alg.data = sfa.get_avalue(ds["BORISOV_2009"])

    # Initialize the network and matrices only once
    #alg.initialize(data=False)

    dfs = []
    vals_alpha = np.linspace(0.1, 0.9, 9)
    for alpha in vals_alpha:
        results = {}
        for abbr, data in ds["BORISOV_2009"].items():
            alg.params.alpha = alpha
            alg.data = data            

            # Do not perform initializing network and matrices multiple times
            alg.initialize()

            alg.compute_batch()
            acc = calc_accuracy(alg.result.df_sim,
                                data.df_exp)

            results[abbr] = acc
        # end of for

        df = pd.DataFrame.from_dict(results, orient='index')
        df.columns = ['alpha_%.1f'%(alpha)]

        dfs.append(df)

    # end of for

    df_merged = pd.concat(dfs, axis=1)
    df_merged = df_merged.sort_index()
    df_merged.to_csv("CPS_borisov_2009_varying_alpha.tsv", sep='\t')

    ind_argmax = np.argmax(df_merged.as_matrix(), axis=1)
    str_argmax = list(df_merged.columns[ind_argmax])
    for i, elem in enumerate(str_argmax):
        print ("%s\t%s"%(df_merged.index[i], elem.rstrip('0')))
    # end of for
            
        
# end of main

