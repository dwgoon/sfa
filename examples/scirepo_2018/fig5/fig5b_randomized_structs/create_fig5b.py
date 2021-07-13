from os.path import join as pjoin

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Data from ODE models (odes)
odes = ['BORISOV_2009',
        'SCHLIEMANN_2011',
        'PEZZE_2012',]

# Data from protein activity (pas)
pas =   ['NELANDER_2008',
         'MOLINELLI_2013',
         'KORKUT_2015A']

dfs = []

for abbr in odes:
    fname = "%s_result_randomized_structs_norm.csv"%(abbr.lower())
    fpath = pjoin("results", fname)
    df = pd.read_csv(fpath, index_col=0)
    udf = df.unstack()
    udf = udf.reset_index(0)
    udf.columns = ['Condition', 'Accuracy']
    udf['Data'] = abbr
    dfs.append(udf)
    print(abbr, udf['Accuracy'].median(), udf['Accuracy'].mean())
    
for abbr in pas:
    fname = "%s_result_randomized_structs_norm.csv"%(abbr.lower())
    fpath = pjoin("results", fname)
    df = pd.read_csv(fpath, index_col=0)
    udf = df.unstack()
    udf = udf.reset_index()
    udf.columns = ['Data', 'Condition', 'Accuracy']
    dfs.append(udf)
    print(abbr, udf['Accuracy'].median(), udf['Accuracy'].mean())
    
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
#sns.set_palette('bright')
sns.set_palette('muted')


#ax = sns.boxplot(data)

colors = sns.color_palette(['orangered',
                            'gold',
                            'yellowgreen',
                            'deepskyblue',
                            'royalblue',
                            'mediumorchid'])
    
#colors = sns.color_palette(n_colors=10)
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
    #artist.set_edgecolor('none')

for line2d in ax.lines:
    line2d.set_color(col)
    line2d.set_mfc(col)
    line2d.set_mec(col)
    line2d.set_solid_capstyle('butt')

# Adjust labels and ticks
ax.set_title('Randomized structure', y=1.1) 
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
plt.subplots_adjust(left=0.2, bottom=0.25,
                    right=0.95, top=0.8,
                    wspace=0.2, hspace=0.2)

plt.savefig('fig5b.png', dpi=600)