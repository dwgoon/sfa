# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:19:45 2016

@author: dwlee
"""

import sfa


if __name__ == "__main__":
    
    algs = sfa.AlgorithmSet()
    algs.create(["PW", "GS", "SP"])

    # Access with the id of algorithm
    alg_pw = algs["PW"]  # Pathway wiring
    alg_gs = algs["GS"]  # Gaussian smoothing
    alg_sp = algs["SP"]  # Signal propagation

    # Print the name of algorithm
    print(alg_pw.name)
    print(alg_gs.name)
    print(alg_sp.name)
       
    # Iterate Algorithms object
    for alg, obj in algs.items():
        print(alg, obj)


    ds = sfa.DataSet()
    ds.create()
    borisov = ds["BORISOV_2009"]["BORISOV_2009_AUC_EGF+I"]

    alg_pw.data = borisov
    alg_gs.data = borisov
    alg_sp.data = borisov

    alg_pw.initialize()
    alg_gs.initialize()
    alg_sp.initialize()

    print()
