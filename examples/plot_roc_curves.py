# -*- coding: utf-8 -*-

import copy
import numpy as np
from scipy import interp

import matplotlib.pyplot as plt

import sfa
    
    
if __name__ == "__main__":
    ds = sfa.DataSet()
    algs = sfa.AlgorithmSet()
    
    data_abbr = "KORKUT_2015"
    class_type = 'UP'
    #data = sfa.get_avalue(ds.create("BORISOV_2009"))
    data = ds.create(data_abbr)
    algs.create(["APS", "NSS", "SP"])

#    algs["NAPS"] = copy.deepcopy(algs["APS"])
#    algs["NAPS"].abbr = "NAPS"
#    algs["NAPS"].params.apply_weight_norm = True
#    
#    algs["NSP"] = copy.deepcopy(algs["SP"])
#    algs["NSP"].abbr = "NSP"
#    algs["NSP"].params.apply_weight_norm = True
    
#    alg_names = ['APS', 'SS', 'SP', 'NAPS', 'NSS', 'NSP']
    
    algs["SS"] = algs["NSS"]
    del algs["NSS"]
    algs["SP"].params.apply_weight_norm = True
    alg_names = ['APS', 'SS', 'SP']
    
    res = {}
    for alg_abbr, alg in algs.items():
        alg.params.use_rel_change = True
        alg.data = data
        alg.initialize()    
        alg.compute_batch()       
    
        fpr, tpr, thr, auroc = sfa.calc_roc_curve(data.df_exp,
                                                  alg.result.df_sim,
                                                  class_type)
        
        
        res[alg_abbr] = (fpr, tpr, auroc)
    

    fig, ax = plt.subplots()
    fig.set_facecolor('white')
    lw = 2
    
    for alg_abbr in alg_names:
        fpr, tpr, auroc = res[alg_abbr]
        plt.plot(fpr["mean"], tpr["mean"],
                 label='{} (area = {:0.2f})'.format(alg_abbr, auroc["mean"]),
                 linewidth=2,)#color='navy', linestyle=':', )
    
#    for i, name_roc in enumerate(data.df_exp.columns):
#        plt.plot(fpr[name_roc], tpr[name_roc], lw=lw,
#                 label='ROC curve of class {0} (area = {1:0.2f})'
#                 ''.format(name_roc, roc_auc[name_roc]))
    # end of for    

    plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
   
        
    ax.set_ylabel("True positive rate")
    ax.set_xlabel("False positive rate")
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
    ax.title.set_fontsize(16)
    ax.tick_params(axis='both', which='major',
                   labelsize=12)
    
    fig.set_size_inches(7, 7)
    fig.savefig('%s_roc_curves_%s.png'%(data_abbr.lower(), class_type),
                facecolor=fig.get_facecolor(),
                dpi=400)            
                #facecolor=fig.get_facecolor(),
                #transparent=True, dpi=400)
