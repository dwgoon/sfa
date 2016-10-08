# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 17:29:59 2016

@author: dwlee
"""


import numpy as np
import sfa


algs = sfa.AlgorithmSet()
algs.create("SP")
alg = algs["SP"]

ds = sfa.DataSet()
ds.create("BORISOV_2009")
data = ds["BORISOV_2009"]["AUC_EGF=1+I=100"]

alg.data = data
alg.initialize()
alg.compute_batch()

df_sim = alg.result.df_sim
df_exp = alg.data.df_exp


res = (np.abs(np.sign(df_sim) - np.sign(df_exp)) == 0).T
#res = np.array(res)

from matplotlib import pyplot as plt
from matplotlib.table import Table

"""
fig = plt.figure(figsize=(15, 4),
                 facecolor='white',
                 edgecolor='k',
                 dpi=100)
heatmap = plt.pcolor(res)

for y in range(res.shape[0]):
    for x in range(res.shape[1]):
        plt.text(x + 0.5, y + 0.5, '$%s$'%('↑'),
                 horizontalalignment='center',
                 verticalalignment='center',)

plt.xlim([0, res.shape[1]])
plt.ylim([0, res.shape[0]])

#plt.colorbar(heatmap)

plt.show()
"""
def checkerboard_table(data, fmt='{:.2f}', bkg_colors=['yellow', 'white']):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0,0,1,1])

    nrows, ncols = data.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i,j), val in np.ndenumerate(data):
        # Index either the first or second item of bkg_colors based on
        # a checker board pattern
        idx = [j % 2, (j + 1) % 2][i % 2]
        color = bkg_colors[idx]

        tb.add_cell(i, j, width, height, text=fmt.format(val), 
                    loc='center', facecolor=color)

    # Row Labels...
    for i, label in enumerate(data.index):
        tb.add_cell(i, -1, width, height, text=label, loc='right', 
                    edgecolor='none', facecolor='none')
    # Column Labels...
    for j, label in enumerate(data.columns):
        tb.add_cell(-1, j, width, height/2, text=label, loc='center', 
                           edgecolor='none', facecolor='none')
    ax.add_table(tb)
    return fig
# end of res   
    

checkerboard_table(res)
plt.show()    
    
    
"""
# Get the perturbed nodes
for i, row in data.df_ba.iterrows():
    ind = np.nonzero(row)[0]
    print (list(row.iloc[ind].index))
    
"""
