from os.path import join as pjoin
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Data from ODE models
odes = ['BORISOV_2009',
        'SCHLIEMANN_2011',
        'PEZZE_2012',]

# Data from protein activity
pas =   ['NELANDER_2008',
         'MOLINELLI_2013',
         'KORKUT_2015A']

droot_data = "results"
fnames = [
    pjoin(droot_data, "%s_result_randomized_weights(-3_0).csv"),
    pjoin(droot_data, "%s_result_randomized_weights_norm(-3_0).csv"),
    pjoin(droot_data, "%s_result_randomized_weights(-3_3).csv"),    
    pjoin(droot_data, "%s_result_randomized_weights_norm(-3_3).csv")
]

# Conditions for weight sampling
conds_weight_samp = ['D', 'N(D)', 'D+A', 'N(D+A)']

dfs = []
for abbr in odes:    
    for i, fname in enumerate(fnames):
        fname = fname%(abbr.lower())
        cond_weight = conds_weight_samp[i]
        df = pd.read_csv(fname, index_col=0)
        udf = df.unstack()
        udf = udf.reset_index(0)
        udf.columns = ['Condition', 'Accuracy']
        items = abbr.split('_')
        year = re.split('\D', items[1])[0]
        udf['Data'] = '%s%s'%(items[0][0], year)
        udf['Weight'] = cond_weight
        dfs.append(udf)
        print(abbr, cond_weight,
              udf['Accuracy'].median(),
              udf['Accuracy'].mean())
    
for abbr in pas:
    for i, fname in enumerate(fnames):
        fname = fname%(abbr.lower())
        cond_weight = conds_weight_samp[i]  
        df = pd.read_csv(fname, index_col=0)
        udf = df.unstack()
        udf = udf.reset_index()
        udf.columns = ['Data', 'Condition', 'Accuracy']
        items = abbr.split('_')
        udf['Data'] = '%s%s'%(items[0][0], items[1][:4])
        udf['Weight'] = cond_weight
        dfs.append(udf)
        print(abbr, cond_weight,
              udf['Accuracy'].median(),
              udf['Accuracy'].mean())  
        
        
        
df_tot = pd.concat(dfs, ignore_index=True)


# Plot
sns.set(font="Arial")
sns_style = {
        'axes.grid': False,
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'font.family': [u'arial'],
        
        }
sns.set_style("ticks", sns_style)
sns.set_palette('muted')

colors = sns.color_palette(['lightsalmon',
                            'orangered',
                            'lightskyblue',
                            'deepskyblue'])

g = sns.FacetGrid(df_tot, col="Data", height=4, aspect=.5,)
g = g.map(sns.boxplot, "Weight", 'Accuracy',
          showfliers=False,
          palette=colors)

g.fig.suptitle('Randomized weight', fontsize=16) 

g.set_titles("{col_name}")
g.set_ylabels("%s"%('Accuracy'), fontsize=14, labelpad=8)

# Set the name of axes
for ax in g.axes[0]:    
    ax.title.set_fontsize(14)
    ax.xaxis.get_label().set_fontsize(12)
    ax.yaxis.get_label().set_fontsize(14)
    
    ax.set_xticklabels(conds_weight_samp, rotation=45)    
    ax.tick_params(axis='x', which='major',
                   labelsize=10, pad=1)    
    ax.tick_params(axis='y', which='major', labelsize=12)
    
    # Set limitations of axes
    plt.ylim([-0.05, 1.0])
    ax.set_xlabel('')

    # Set the colors of lines
    col = 'black'
    
    for i, artist in enumerate(ax.artists):
        artist.set_linewidth(1)
        artist.set_edgecolor('black')
    
    for line2d in ax.lines:
        line2d.set_color(col)
        line2d.set_mfc(col)
        line2d.set_mec(col)
        line2d.set_solid_capstyle('butt')

    
g.fig.subplots_adjust(left=0.09, bottom=0.2,
                      right=0.95, top=0.8,
                      wspace=0.1, hspace=0.2)

g.fig.set_size_inches(9.54, 3.5)
plt.savefig('fig5c.png', dpi=600)


