# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.table import Table

import numpy as np
import pandas as pd

mpl.rc('font', family='Arial')
plt.ioff()  # Suppress showing figures


import sfa

def get_label_fontsize(label, fsz, wc=3):
    """
    label: label string
    fsz: the default font size
    wc: the count of words corresponding to default font size
    """
    l = len(label)
    if l>wc:
        return 1.2*fsz*wc/float(l)
    else:
        return fsz
        

def plot_table(dfc, dfr, dfe, font_sz=4):
    fig, ax = plt.subplots()
    
    ax.set_frame_on(False)
    ax.get_yaxis().set_visible(False)
    tb = Table(ax, bbox=[0,0,1,1])
    tb.auto_set_font_size(False)

    nrows = dfc.shape[0]
    ncols = dfc.shape[1] + dfr.shape[1] + 1
    
    L = 1.0
    H = 1.0
    w_stgap = 0.01 # sub-table gap width    
    
    width = (L - w_stgap)/(ncols-1)
    height = H / nrows

    w_tgap = width  # tick gap width
    
    """
    n_cn: the number of nodes used in perturbation conditions
    n_rn: the number of readout nodes
    """
    n_conds, n_cn = dfc.shape
    _, n_rn = dfr.shape
    
    # Add vertical gap
    for i in range(-1, n_conds):
        tb.add_cell(i, n_cn, w_stgap, height, 
                    text='',
                    loc='center',
                    edgecolor='none',
                    facecolor='none')
    
    # Experiemntal conditions for perturbation
    for (i,j), val in np.ndenumerate(dfc):
        if val != 0:
            fcolor = 'blue'
        else:
            fcolor = 'white'

        tb.add_cell(i, j, width, height, 
                    text='',
                    loc='center',
                    facecolor=fcolor)
        
    # Agreement between the results of simulation and experiment
    for (i,j), val_sim in np.ndenumerate(dfr):
        val_exp = dfe.iloc[i,j]
        if val_sim>0:
            fcolor = '#35FF50'            
        else:
            fcolor = '#FF0000'
            
        if val_exp>0:            
            text_arrow = 'UP'
        elif val_exp<0:
            text_arrow = 'DN'
        else:
            text_arrow = '─'

        tb.add_cell(i, n_cn+1+j, width, height, 
                    text=text_arrow,
                    loc='center',
                    facecolor=fcolor)
    # end of for
                    
    # Adjust font size and color of the result
    celld = tb.get_celld()
    for (i,j), val in np.ndenumerate(dfr):
        cell = celld[(i,n_cn+1+j)]
        
        if val>0:
            tcolor = '#000000'
        else:
            tcolor = '#FFFFFF'
        # end of if-else
        
        cell.set_text_props(color=tcolor,
                            #fontsize=5,
                            weight='bold')
    # end of for

    # Row Labels
    for i, label in enumerate(dfr.index):
        tb.add_cell(i, -1, 1.5*width, height, text=label, loc='right', 
                    edgecolor='none', facecolor='none',)
                    
    
    # Resize text fonts
    tb.set_fontsize(font_sz)


    # Adjust the width of table lines
    celld = tb.get_celld()
    for (i,j), cell in celld.items():    
        cell.set_linewidth(0.5)
    
    # Add table graphic object
    ax.add_table(tb)        
        
    # Add column labels using x-axis
    xlabels = list(dfc.columns) + [''] + list(dfr.columns)
    ax.xaxis.tick_top()    


    xticks = [width/2.0,]
    for j in range(1, n_cn):
        xticks.append(xticks[j-1]+w_tgap)

    xticks.append(xticks[n_cn-1] + width/2.0  + w_stgap/2.0)        
    xticks.append(xticks[n_cn] + width/2.0  + w_stgap/2.0)
    
    for j in range(1, n_rn):
        xticks.append(xticks[n_cn+j]+w_tgap)
        
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=90, minor=False)
    #ax.grid(False)

    # Adjust the label size
    ax.tick_params(axis='x', which='major', labelsize=5)    
    
    
    # Hide the small bars of ticks
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1On = False
        tick.tick2On = False

    return fig, tb

if __name__ == '__main__':

    algs = sfa.AlgorithmSet()    
    algs.create("SP")
    alg = algs["SP"]
    
    ds = sfa.DataSet()
    ds.create("BORISOV_2009")
    data = ds["BORISOV_2009"]["AUC_EGF=1+I=100"]

    alg.params.is_rel_change = True
    alg.data = data
    alg.initialize()
    alg.compute_batch()
    
    dfc = alg.data.df_ba

    # Get the perturbed nodes
    df_sim = alg.result.df_sim
    df_exp = alg.data.df_exp
    dfr = (np.abs(np.sign(df_sim) - np.sign(df_exp)) == 0)
    dfe = alg.data.df_exp
    
    fig, tb = plot_table(dfc, dfr, dfe)
    fig.set_facecolor('white')
    fig.set_size_inches(3, 8)
    fig.savefig("test.png", dpi=300)
    plt.close(fig)    
    

"""
[References]
http://stackoverflow.com/questions/10194482/custom-matplotlib-plot-chess-board-like-table-with-colored-cells
http://stackoverflow.com/questions/14391959/heatmap-in-matplotlib-with-pcolor
"""