# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import glob


abbrs = ["BORISOV_2009", "SCHLIEMANN_2011", "PEZZE_2012"]

dfs = []
#for fname in glob.glob("*_acc.tsv"):
for data_abbr in abbrs:
    #fname = "algs_nelander_2008_acc.tsv"
    #items = fname.split('_')
    #data_abbr = '_'.join(items[1:3])
    #data_abbr = data_abbr.upper()
    fname = "algs_%s_acc.tsv"%(data_abbr.lower())
    df = pd.read_table(fname, index_col=0)
    df = df.unstack().reset_index()
    df.columns = ['Algorithm', 'Condition', 'Accuracy']
    df['Data'] = data_abbr
    dfs.append(df)
# ax = sns.boxplot(df[['APS', 'SS', 'SP']])
df_tot = pd.concat(dfs, axis=0, ignore_index=True)

#ax = sns.boxplot(data)
ax = sns.boxplot(x='Data', y='Accuracy', hue='Algorithm', data=df_tot)
 
# Set the name of axes
ax.set_ylabel("Accuracy")
ax.set_xlabel("Algorithms")
ax.xaxis.get_label().set_fontsize(16)
ax.yaxis.get_label().set_fontsize(16)

ax.tick_params(axis='x', which='major', labelsize=14)
ax.tick_params(axis='y', which='major', labelsize=14)



plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9,
                    wspace=0.2, hspace=0.2)