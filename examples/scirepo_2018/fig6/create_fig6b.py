from functools import reduce
from operator import add

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sfa
from sfa.plot import HierarchicalClusteringTable


if __name__ == '__main__':
    algs = sfa.AlgorithmSet()    
    algs.create("SP")
    alg = algs["SP"]
    
    ds = sfa.DataSet()
    mdata = ds.create("BORISOV_2009")

    alg.params.alpha = 0.5
    alg.params.use_rel_change = True
    alg.params.apply_weight_norm = True
    
    list_abbr = []
    list_acc = []
    list_cons = []
    
    for abbr, data in mdata.items():
        
        alg.data = data
        alg.initialize()
        
        n2i = data.n2i        
        alg.compute_batch()

        # Get the perturbed nodes
        acc, cons = sfa.calc_accuracy(alg.data.df_exp,
                                      alg.result.df_sim,
                                      get_cons=True)
        list_abbr.append(abbr)
        list_acc.append(acc)
        list_cons.append(pd.DataFrame(cons, dtype=np.int16))
    # end of for 
    
    avg_cons = reduce(add, list_cons)/len(list_cons)

    # Plot
    width_ratios = [1, 0.05, 1, 0.5, 0.1]
    height_ratios = [0.3, 1]
    hct = HierarchicalClusteringTable(data.df_conds, avg_cons,
                                      col_metric='cityblock',
                                      row_metric='cityblock',
                                      vmin = 0.0,
                                      vmax = 1.0,
                                      wspace=0.015,
                                      width_ratios=width_ratios,
                                      height_ratios=height_ratios,
                                      table_tick_fontsize=5,
                                      colorbar_tick_fontsize=5,
                                      row_dend_linewidth=0.25,
                                      col_dend_linewidth=0.25,
                                      table_linewidth=0.25)


    # Adjust figure size
    hct.fig.set_size_inches(3.5, 3.5)
    hct.axes['condition'].set_xlabel('Targets', size=8, labelpad=5)
    hct.axes['heatmap'].set_xlabel('Readouts', size=8, labelpad=5)
    hct.colorbar.set_label('Average accuracy',
                           rotation=270,
                           labelpad=12,
                           size=8)
    
    plt.subplots_adjust(left=0.05, bottom=0.18,
                        right=0.88, top=0.98,
                        wspace=0.2, hspace=0.2)


    hct.fig.savefig("fig6b.png", dpi=1200)
    
    mean_acc_sorted = avg_cons.mean(axis=1).sort_values()
