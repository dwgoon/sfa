# -*- coding: utf-8 -*-

import copy
import pandas as pd

import sfa
import sfa.vis
from sfa import calc_accuracy
from sfa import AlgorithmSet
from sfa import DataSet


if __name__ == "__main__":

    # Create containers for algorithm and data.
    algs = AlgorithmSet()
    algs.create(['CPS', 'SS', 'NSS', 'SP'])

    ds = DataSet()

    ds.create("KORKUT_2015")
    data = ds["KORKUT_2015"]



    # Normalized PS
    # algs["NAPS"] = copy.deepcopy(algs["APS"])
    # algs["NAPS"].abbr = "NAPS"
    # algs["NAPS"].params.apply_weight_norm = True
    #
    algs["NCPS"] = copy.deepcopy(algs["CPS"])
    algs["NCPS"].abbr = "NCPS"
    algs["NCPS"].params.apply_weight_norm = True

    algs["NSP"] = copy.deepcopy(algs["SP"])
    algs["NSP"].params.apply_weight_norm = True
    algs["NSP"].abbr = "NSP"


    results = {}
    for abbr, alg in algs.items():        
        alg.params.use_rel_change = True
        alg.data = data
        alg.initialize()

        alg.compute_batch()
        df_sim = alg.result.df_sim
        acc, cons = calc_accuracy(df_sim,
                                  data.df_exp,
                                  get_cons=True)

        results[abbr] = acc
        print ("%s: %.3f"%(abbr, acc))


        #brt = sfa.vis.BatchResultTable(data, cons)
        #brt.fig.set_size_inches(4, 6)
        #brt.fig.savefig("%s_%s.png" % (alg.abbr, data.abbr), dpi=300)
    # end of for
        
    df = pd.DataFrame.from_dict(results, orient='index')
    df.columns = [data.abbr]
    #df = df.ix[["APS", "NAPS", "CPS", "NCPS", "SS", "NSS", "SP", "NSP"], :]
    df = df.ix[["CPS", "NCPS", "SS", "NSS", "SP", "NSP"], :]
    print(df)

    #df_sort = df.sort_values(by='MOLINELLI_2013')
    #print(df_sort)


# end of main
