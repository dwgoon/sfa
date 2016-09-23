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
    alg_sp = algs["SP"]

    ds.create("BORISOV_2009")

    # Make a sequence of data using a list
    borisov = []

    borisov.append("AUC_EGF=0.1+I=100")
    borisov.append("AUC_EGF=1+I=100")
    borisov.append("AUC_EGF=20+I=100")

    alg_sp.params.is_rel_change = True
    alg_sp.data = ds["BORISOV_2009"]["AUC_EGF=0.1+I=100"]
    alg_sp.initialize(init_data=False)

    #for abbr in borisov:
    for abbr, data in ds["BORISOV_2009"].items():
        #data = ds["BORISOV_2009"][abbr]
        alg_sp.data = data
        alg_sp.initialize(init_network=False)
        alg_sp.compute_batch()
        acc = calc_accuracy(alg_sp.result.df_sim,
                            data.df_exp)

        print("Accuracy for %s: " % (abbr), acc)
    # end of for