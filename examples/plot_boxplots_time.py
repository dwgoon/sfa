# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class_type = 'UP' # Class label: UP, DN, -
measure = 'Time'
abbrs = ["BORISOV_2009", "SCHLIEMANN_2011", "PEZZE_2012"]

dfs = []
for data_abbr in abbrs:
    fname = "algs_%s_%s.tsv"%(data_abbr.lower(), measure.lower(), )
    df = pd.read_table(fname, index_col=0)
    df = df.unstack().reset_index()
    df.columns = ['Algorithm', 'Condition', measure]
    df['Data'] = data_abbr
    dfs.append(df)

    df_tot = pd.concat(dfs, axis=0, ignore_index=True)

#ax = sns.boxplot(data)
#ax = sns.boxplot(x='Data',
#                 y=measure,
#                 hue='Algorithm',
#                 data=df_tot,
#                 palette='Set2',)
 
#sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set(font_scale=1.11)
g = sns.FacetGrid(df_tot, col="Data", size=4, aspect=.5, sharey=False)
g = g.map(sns.boxplot, "Algorithm", measure,  palette="Set3")
g.set_ylabels('Time (sec.)')

# Set the name of axes
#ax.set_ylabel(ylabel)
#ax.set_xlabel("Algorithms")
for ax in g.axes[0]:
    ax.xaxis.get_label().set_fontsize(12)
    ax.yaxis.get_label().set_fontsize(14)
    
    ax.tick_params(axis='x', which='major', labelsize=12)
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.set_yscale("log", nonposy='clip')
    
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9,
                    wspace=0.25, hspace=0.2)