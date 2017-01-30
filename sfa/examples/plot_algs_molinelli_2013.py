# -*- coding: utf-8 -*-

import copy
import pandas as pd
import matplotlib.pyplot as plt

import sfa
import sfa.vis
from sfa import calc_accuracy
from sfa import AlgorithmSet
from sfa import DataSet

plt.ioff()

if __name__ == "__main__":

    # Create containers for algorithm and data.
    algs = AlgorithmSet()
    ds = DataSet()

    # Load an algorithm and a data.
    #algs.create(["CPS", "SP"])
    algs.create()

    algs["APS"] = algs["APS"]
    algs["APS"].abbr = "APS"
    
    # Normalized CPS
    algs["NCPS"] = copy.deepcopy(algs["CPS"])
    algs["NCPS"].abbr = "NCPS"
    algs["NCPS"].params.apply_weight_norm = True

    algs["NAPS"] = copy.deepcopy(algs["APS"])
    algs["NAPS"].abbr = "NAPS"
    algs["NAPS"].params.initialize()
    algs["NAPS"].params.apply_weight_norm = True

    ds.create("MOLINELLI_2013")
    data = ds["MOLINELLI_2013"]
    
    #data.df_conds.drop(data.df_conds.columns[:17], axis=1, inplace=True)
    #data.df_exp.drop(data.df_exp.columns[17:], axis=1, inplace=True)
    #data.df_exp = data.df_exp[['cellprol']]
        
    
    alpha = 0.9
    algs["SS"].params.alpha = alpha
    algs["NSS"].params.alpha = alpha
    
    dfs = {}
    results = {}
    for abbr, alg in algs.items():        
        alg.params.use_rel_change = True
        #alg.params.exsol_forbidden = True
        alg.data = data
        alg.initialize()
        
        #W = sfa.normalize(data.A, norm_in=False)
        #alg.W = W
        
        alg.compute_batch()
        #df_sim = alg.result.df_sim.ix[:, :17]
        #df_sim = alg.result.df_sim[['cellprol']]
        df_sim = alg.result.df_sim

        acc, cons = calc_accuracy(df_sim,
                                  data.df_exp,
                                  get_cons=True)

        brt = sfa.vis.BatchResultTable(data, cons)
        brt.fig.set_size_inches(4, 6)
        brt.fig.savefig("%s_%s.png"%(alg.abbr, data.abbr), dpi=300)
        #brt.fig.savefig("output_%s_%s.png"%(alg.abbr, data.abbr), dpi=300)
    
        results[abbr] = acc
        print ("%s: %.3f"%(abbr, acc))
        dfs[abbr] = df_sim
    # end of for
        

#    df = pd.DataFrame.from_dict(results, orient='index')
#    df.columns = ['MOLINELLI_2013']
#    
#    df_sort = df.sort_values(by='MOLINELLI_2013')
#    print(df_sort)
    plt.close("all")
    
    #df_diff = (dfs["SP"] == dfs["NCPS"])

#    ht = sfa.vis.HeatmapTable()
#    ht.table_fontsize = 10
#    ht.fig.set_size_inches(6, 8)
#    plt.subplots_adjust(left=0.4, bottom=0.05, right=0.9, top=0.9,
#                        wspace=0.4, hspace=0.2)
#
#    ht.fig.savefig("SP_vs_NCPS.png", dpi=300)    

# end of main
