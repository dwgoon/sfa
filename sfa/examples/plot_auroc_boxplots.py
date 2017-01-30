# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

fname = "algs_nelander_2008_rocauc_dn.tsv"
df = pd.read_table(fname, index_col=0)

# ax = sns.boxplot(df[['APS', 'SS', 'SP']])

ax = sns.boxplot(data=df)

 # Set the name of axes
ax.set_ylabel("AUROC")
ax.set_xlabel("Algorithms")
ax.xaxis.get_label().set_fontsize(16)
ax.yaxis.get_label().set_fontsize(16)

ax.tick_params(axis='x', which='major', labelsize=14)
ax.tick_params(axis='y', which='major', labelsize=14)


plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9,
                    wspace=0.2, hspace=0.2)