# -*- coding: utf-8 -*-

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

    ds.create("NELENDER_2008")

    # Set the loaded data as the data of the algorithm
    alg_sp = algs["SP"]
    """
    data = ds["NELENDER_2008"]
    alg_sp.data = data

    # Initialize the algorithm
    alg_sp.initialize()

    # Perform computation
    alg_sp.compute()

    # Fetch the result from the algorithm
    res = alg_sp.result

    acc = calc_accuracy(res.df_sim, data.df_exp)
    print("Accuracy for %s: "%(data.abbr), acc)
    """

    ds.create("BORISOV_2009")

    borisov = []
    borisov.append("BORISOV_2009_AUC_CTRL")
    borisov.append("BORISOV_2009_AUC_EGF")
    borisov.append("BORISOV_2009_AUC_I")
    borisov.append("BORISOV_2009_AUC_EGF+I")
    borisov.append("BORISOV_2009_SS_CTRL")
    borisov.append("BORISOV_2009_SS_EGF")
    borisov.append("BORISOV_2009_SS_I")
    borisov.append("BORISOV_2009_SS_EGF+I")

    alg_sp.params.is_rel_change = True
    for abbr in borisov:
        data = ds["BORISOV_2009"][abbr]
        alg_sp.data = data
        alg_sp.initialize()
        alg_sp.compute()
        acc = calc_accuracy(alg_sp.result.df_sim,
                            data.df_exp)

        print("Accuracy for %s: " % (abbr), acc)
    # end of for