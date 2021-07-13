import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from sfa import calc_accuracy
from sfa import AlgorithmSet
from sfa import DataSet


# Create data structures to collect the results.
dfs = []

# Datasets of ODE models
odes = ['BORISOV_2009',
        'SCHLIEMANN_2011',
        'PEZZE_2012',]

# Datasets of perturbation biology
pas = ['NELANDER_2008',
       'MOLINELLI_2013',
       'KORKUT_2015A']

# Load the results of ODE datasets.
for abbr in odes:
    fname = "algs_%s_accuracy.tsv"%(abbr.lower())
    df = pd.read_table(fname, index_col=0)
    df = df.unstack().reset_index()
    df.columns = ['Algorithm', 'Condition', 'Accuracy']
    del df['Algorithm']  # Remove algorithm column for concatenation
    df['Data'] = abbr
    print(abbr, df['Accuracy'].median(), df['Accuracy'].mean(), df['Accuracy'].var())
    dfs.append(df)


# Create containers for algorithm and data.
algs = AlgorithmSet()
ds = DataSet()

# Load an algorithm and a data.
algs.create(['SP'])
alg_names = ['SP']
   
ds.create(["NELANDER_2008", "MOLINELLI_2013", "KORKUT_2015A"])

alg = algs['SP']
alg.params.apply_weight_norm = True
alg.params.use_rel_change = True
alg.params.alpha = 0.5

# Directly obtain the results of perturbation biology datasets.
for abbr, data in ds.items():
    alg.data = data        
    alg.initialize()
    alg.compute_batch()
    acc, cons = calc_accuracy(alg.result.df_sim,
                              data.df_exp,
                              get_cons=True)

    df = pd.DataFrame(cons.mean(axis=1))
    udf = df.unstack()
    udf = udf.reset_index(1)
    udf.columns = ['Condition', 'Accuracy']
    udf['Data'] = abbr
    dfs.append(udf)
    print (abbr, udf['Accuracy'].median(), udf['Accuracy'].mean(), udf['Accuracy'].var())
    
# end of for
    

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

colors = sns.color_palette(['orangered',
                            'gold',
                            'yellowgreen',
                            'deepskyblue',
                            'royalblue',
                            'mediumorchid'])
    

ax = sns.boxplot(x='Data', y='Accuracy',
             data=df_tot,
             orient='v',
             palette=colors,
             showfliers=False)
 
# Set the name of axes
ax.set_ylabel("Accuracy", color='black', labelpad=8)
ax.set_xlabel("Data", color='black', labelpad=8)
ax.set_xticklabels(['B2009', 'S2011', 'P2012', 'N2008', 'M2013', 'K2015'])


# Disable the top and right axes
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)


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

# Adjust labels and ticks
ax.set_title('Original', y=1.1) 
ax.title.set_fontsize(16)

ax.set_xticklabels(ax.get_xticklabels(), rotation=0) 
ax.xaxis.get_label().set_fontsize(14)
ax.yaxis.get_label().set_fontsize(14)

ax.tick_params(axis='x', which='major',
               labelsize=12, labelcolor='black')
ax.tick_params(axis='y', which='major',
               labelsize=12, labelcolor='black')

# Set limitations of axes
plt.ylim([0.0, 1.0])


fig = ax.figure
fig.set_size_inches(4.8, 3.2)
plt.subplots_adjust(left=0.2,
                    bottom=0.25,
                    right=0.95,
                    top=0.8,
                    wspace=0.2,
                    hspace=0.2)


plt.savefig('fig5a.png', dpi=600)
