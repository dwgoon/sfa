# -*- coding: utf-8 -*-

import copy
import numpy as np
import pandas as pd

import sfa
import sfa.plot
from sfa import calc_accuracy
from sfa import AlgorithmSet
from sfa import DataSet


if __name__ == "__main__":

    # Create containers for algorithm and data.
    algs = AlgorithmSet()
    algs.create(['NSS', 'SP'])

    ds = DataSet()

    ds.create("KORKUT_2015")
    data = ds["KORKUT_2015"]
    algs["SP"].params.apply_weight_norm = True
    
    results = {}
    for abbr, alg in algs.items():        
        alg.params.use_rel_change = True
        alg.data = data
        alg.initialize()

        alg.compute_batch()
        df_sim = pd.DataFrame(alg.result.df_sim)            
#        if abbr == 'SP':
#            thr_up = 0 #2.130000e-10
#            thr_dn = -1.960000e-10
#            
#        elif abbr == 'NSS':
#            thr_up = 1.731947e-05
#            thr_dn = -1.708759e-05
#            
#        df_sim[df_sim>thr_up] = 1.0
#        df_sim[df_sim<thr_dn] = -1.0
#        df_sim[np.logical_and(df_sim>thr_dn, df_sim<thr_up)] = 0.0
        
        acc = sfa.calc_accuracy(data.df_exp, df_sim)
            
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
    df = df.ix[["NSS", "SP"], :]
    print(df)

    #df_sort = df.sort_values(by='MOLINELLI_2013')
    #print(df_sort)


# end of main
