# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import sfa
from sfa.vis import BatchResultTable

if __name__ == '__main__':

    algs = sfa.AlgorithmSet()    
    algs.create("SP")
    alg = algs["SP"]
    
    ds = sfa.DataSet()
    #ds.create("BORISOV_2009")
    #data = ds["BORISOV_2009"]["AUC_EGF=1+I=100"]
    #ds.create("NELANDER_2008")
    #data = ds["NELANDER_2008"]
    ds.create("MOLINELLI_2013")
    data = ds["MOLINELLI_2013"]

    alg.params.use_rel_change = True
    alg.data = data
    alg.initialize()
    alg.compute_batch()

    # Get the perturbed nodes
    acc, cons = sfa.calc_accuracy(alg.data.df_exp,
                                  alg.result.df_sim,
                                  get_cons=True)

    brt = BatchResultTable(alg.data, cons)
    #brt.fig.set_size_inches(4, 8)
    brt.fig.set_size_inches(6, 6)
    brt.fig.savefig("BORISOV_2009_AUC_EGF=1+I=100.png", dpi=300)
    plt.show()