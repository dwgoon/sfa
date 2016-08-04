# -*- coding: utf-8 -*-
"""
@author: dwlee
"""



import sfa
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
    alg_sp.data = ds["NELENDER_2008"]

    # Initialize the algorithm
    alg_sp.initialize()

    # Perform computation
    alg_sp.compute()

    # Fetch the result from the algorithm
    res = alg_sp.result

