# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class_type = 'DN' # Class label: UP, DN, -
ylabel = "AUPRC"
abbrs = ["BORISOV_2009", "SCHLIEMANN_2011", "PEZZE_2012"]
#abbrs = ["BORISOV_2009",]

dfs = []
#for fname in glob.glob("*_acc.tsv"):
for data_abbr in abbrs:
    #fname = "algs_nelander_2008_acc.tsv"
    #items = fname.split('_')
    #data_abbr = '_'.join(items[1:3])
    #data_abbr = data_abbr.upper()
    #fname = "algs_%s_acc.tsv"%(data_abbr.lower())
    if ylabel == 'Accuracy':
        fname = "algs_%s_%s.tsv"%(data_abbr.lower(), ylabel.lower(), )
    elif ylabel == 'AUROC' or ylabel == 'AUPRC':        
        fname = "algs_%s_%s_%s.tsv"%(data_abbr.lower(),
                                     ylabel.lower(),
                                     class_type.lower())
        
    df = pd.read_table(fname, index_col=0)
    
    df = df[['APS', 'NSS', 'NSP']]
    df.columns = df.columns.str.replace('NSS', 'SS')
    df.columns = df.columns.str.replace('NSP', 'SP')
    
    df = df.unstack().reset_index()
    df.columns = ['Algorithm', 'Condition', ylabel]
    df['Data'] = data_abbr
    dfs.append(df)
# ax = sns.boxplot(df[['APS', 'SS', 'SP']])
df_tot = pd.concat(dfs, axis=0, ignore_index=True)

#ax = sns.boxplot(data)
#ax = sns.boxplot(x='Data',
#                 y=ylabel,
#                 hue='Algorithm',
#                 data=df_tot,
#                 palette='Set2',)
 
#sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set(font_scale=1.11)
g = sns.FacetGrid(df_tot, col="Data", size=4, aspect=.5,)
g = g.map(sns.boxplot, "Algorithm", ylabel,  palette="Set3")
g.set_titles("{col_name}")
g.set_ylabels("%s (%s)"%(ylabel, class_type))
# Set the name of axes
#ax.set_ylabel(ylabel)
#ax.set_xlabel("Algorithms")
for ax in g.axes[0]:
    ax.title.set_fontsize(12)
    ax.xaxis.get_label().set_fontsize(12)
    ax.yaxis.get_label().set_fontsize(16)
    
    ax.tick_params(axis='x', which='major', labelsize=12)
    ax.tick_params(axis='y', which='major', labelsize=12)
    
plt.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.9,
                    wspace=0.1, hspace=0.2)


plt.show()

g.fig.set_size_inches(7, 7)

str_datasets = '_'.join([abbr.lower() for abbr in abbrs])
if ylabel == 'Accuracy':
    g.fig.savefig('accuracy_%s.png'%(str_datasets), dpi=400)
elif ylabel == 'AUROC' or ylabel == 'AUPRC':        
    g.fig.savefig('%s_%s_%s.png'%(ylabel.lower(),
                                  class_type.lower(),
                                  str_datasets),
                  dpi=400)
          #transparent=True, dpi=400)

