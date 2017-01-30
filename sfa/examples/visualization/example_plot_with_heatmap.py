# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sfa
from sfa.vis import HeatmapTable

if __name__ == '__main__':

    algs = sfa.AlgorithmSet()    
    algs.create("SP")
    alg = algs["SP"]
    
    ds = sfa.DataSet()
    ds.create("BORISOV_2009")

    alg.params.use_rel_change = True
    list_abbr = []
    list_acc = []
    for abbr, data in ds["BORISOV_2009"].items():
        alg.data = data
        alg.initialize()
        alg.compute_batch()

        # Get the perturbed nodes
        acc, cons = sfa.calc_accuracy(alg.data.df_exp,
                                      alg.result.df_sim,
                                      get_cons=True)
        list_abbr.append(abbr)
        list_acc.append(acc)
    # end of for

    # Prepare a DataFrame for plotting
    arr = np.tile(np.array(list_acc), reps=(5,1)).T
    df = pd.DataFrame(data=arr)
    df.columns = ['SP1', 'SP2', 'SP3', 'SP4', 'SP5']
    df.index = list_abbr

    # Draw the heatmap with the prepared DataFrame
    ht = HeatmapTable(df, cmap='hot')
    ht.table_fontsize = 10
    ht.fig.set_size_inches(6, 8)
    plt.subplots_adjust(left=0.4, bottom=0.05, right=0.9, top=0.9,
                        wspace=0.4, hspace=0.2)

    # Set the name of axes
    ht.ax.set_ylabel("Algorithms")
    ht.ax.set_xlabel("Data")
    ht.ax.xaxis.get_label().set_fontsize(16)
    ht.ax.yaxis.get_label().set_fontsize(16)

    # Save the image file
    ht.fig.savefig("accuracy_algs_vs_ds.png", dpi=300)
    plt.show()
