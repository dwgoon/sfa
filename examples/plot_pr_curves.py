# -*- coding: utf-8 -*-

import copy
import numpy as np
import scipy as sp
from scipy import interp

import matplotlib.pyplot as plt

import sfa
    
    
if __name__ == "__main__":
    ds = sfa.DataSet()
    algs = sfa.AlgorithmSet()
    
    data_abbr = "MOLINELLI_2013"
    class_type = 'UP'
    data = ds.create(data_abbr)
    algs.create(["APS", "SS", "NSS", "SP"])

    algs["NAPS"] = copy.deepcopy(algs["APS"])
    algs["NAPS"].abbr = "NAPS"
    algs["NAPS"].params.apply_weight_norm = True
    
    algs["NSP"] = copy.deepcopy(algs["SP"])
    algs["NSP"].abbr = "NSP"
    algs["NSP"].params.apply_weight_norm = True
    
    alg_names = ['APS', 'SS', 'SP', 'NAPS', 'NSS', 'NSP']
    
    res = {}
    for alg_abbr, alg in algs.items():
        alg.params.use_rel_change = True
        alg.data = data
        alg.initialize()    
        alg.compute_batch()       
    
        recall, precision, thr, auprc \
            = sfa.calc_pr_curve(data.df_exp, alg.result.df_sim, class_type)
        
        res[alg_abbr] = (recall, precision, auprc)
    # end of for

    fig, ax = plt.subplots()
    fig.set_facecolor('white')
    lw = 2
    
    for alg_abbr in alg_names:
        recall, precision, auprc = res[alg_abbr]
        plt.plot(recall["mean"], precision["mean"],
                 label='{} (area = {:0.2f})'.format(alg_abbr, auprc["mean"]),
                 linewidth=2,)#color='navy', linestyle=':', )
    

    # Base-line
    plt.plot([0, 1], [0.5, 0.5], color='red', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.legend(loc="lower right")
    plt.show()
   
    ax.set_title('Precision-Recall (PR)')
    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    ax.title.set_fontsize(16)
    ax.tick_params(axis='both', which='major',
                   labelsize=12)
    
    fig.set_size_inches(7, 7)
    fig.savefig('%s_prc_curves_%s.png'%(data_abbr.lower(), class_type),
                facecolor=fig.get_facecolor(),
                transparent=True, dpi=400)

#    recall, precision, auprc = res["SP"]
#    for name in recall:
#        plt.plot(recall[name], precision[name],
#         label='{} (area = {:0.2f})'.format(name, auprc[name]),
#         linewidth=2,)
#        
#    plt.legend(loc="lower right")
#    all_recall = np.unique(np.concatenate([rec for name, rec in recall.items()]))
#    mean_precision = np.zeros_like(all_recall)
#    for name in recall:
#        mean_precision += sp.interp(all_recall,
#                                    recall[name],
#                                    precision[name])
#        
        
    
