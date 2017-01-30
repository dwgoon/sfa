# -*- coding: utf-8 -*-

import sfa


if __name__ == "__main__":

    """
    algs = sfa.AlgorithmSet()
    algs.create("APS")
    algs.create("SS")
    algs.create("SP")

    The following code snippet is the same as the above.

    AlgorithmSet's create() without any argument creates all algorithms.
    """

    algs = sfa.AlgorithmSet()
    algs.create(["APS", "SS", "SP"])


    # Access with the id of algorithm
    alg_aps = algs["APS"]  # Pathway wiring
    alg_gs = algs["SS"]  # Signal smoothing
    alg_sp = algs["SP"]  # Signal propagation

    # Print the name of algorithm
    print(alg_aps.name)
    print(alg_gs.name)
    print(alg_sp.name)
       
    # Iterate Algorithms object
    for alg, obj in algs.items():
        print(alg, obj)




