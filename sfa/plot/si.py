
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

def siplot(df_splo,
           df_inf,
           output,
           min_splo=None,
           max_splo=None,
           thr_inf=1e-10,
           fmt_inf='%f',
           fig=None,
           cnt_max=None,
           ncol=4,
           designated=None,
           color='silver',
           dcolor='red',
           zcolor='red',
           alpha=0.7,
           xfontsize=8,
           yfontsize=8):

    # SPLO-Influence Data
    if not min_splo:
        min_splo = df_splo.min()

    if not max_splo:
        max_splo = df_splo.max()

    mask_splo = (min_splo <= df_splo) & (df_splo <= max_splo)
    df_splo = df_splo[mask_splo]

    df_splo = pd.DataFrame(df_splo)
    df_splo.columns = ['SPLO']


    if output in df_splo.index:
        df_splo.drop(output, inplace=True)

    index_common = df_splo.index.intersection(df_inf.index)
    df_inf = pd.DataFrame(df_inf.loc[index_common])

    mark_drop = df_inf[output].abs() <= thr_inf
    df_inf.drop(df_inf.loc[mark_drop, output].index,
                inplace=True)


    df_si = df_inf.join(df_splo.loc[index_common])
    df_si.index.name = 'Source'
    df_si.reset_index(inplace=True)

    cnt_splo = Counter(df_si['SPLO'])
    if not cnt_max:
        cnt_max = max(cnt_splo.values())

    splos = sorted(cnt_splo.keys())
    nrow = int(np.ceil(len(splos)/ncol))

    # Plot
    if not fig:
        fig = plt.figure()

    gs = gridspec.GridSpec(nrow, ncol)

    yvals = np.arange(1, cnt_max +1)
    for i, splo in enumerate(splos):
        idx_row = int(i / ncol)
        idx_col = int(i % ncol)
        ax = fig.add_subplot(gs[idx_row, idx_col])
        df_sub = df_si[df_si['SPLO'] == splo]
        df_sub = df_sub.sort_values(by=output)
        num_items = df_sub[output].count()

        influence = np.zeros((cnt_max,))  # Influence
        num_empty = cnt_max - num_items
        influence[num_empty:] = df_sub[output]
        names = df_sub['Source'].tolist()
        names = ['' ] *(num_empty) + names

        # Plot bars
        plt.barh(yvals, influence, align='center',
                 alpha=alpha)

        ax.set_title('SPLO=%d'%(splo))
        ax.set_xlabel('')

        ax.xaxis.set_major_formatter(FormatStrFormatter(fmt_inf))
        ax.tick_params(axis='x',
                       which='major',
                       labelsize=xfontsize)

        ax.set_ylabel('')
        ax.yaxis.set_ticks_position('right')
        ax.tick_params(axis='y',
                       which='major',
                       labelsize=yfontsize)

        plt.yticks(yvals, names)

        # Draw zero line.
        if not((influence <= 0).all() or (influence >= 0).all()):
            ax.vlines(x=0.0, ymin=0, ymax=yvals[-1]+1, color=zcolor)

        # Set limitations
        ax.set_ylim(0, cnt_max +1)
        
        # Filter bar graphics.
        bars = []
        cnt_bars = 0
        for obj in ax.get_children():
            if cnt_bars == cnt_max:
                break
            if isinstance(obj, Rectangle):
                bars.append(obj)
                obj.set_color(color)
                cnt_bars += 1
        # end of for
        
        if designated:
            # Change the bars of the designated names.
            for i, name in enumerate(names):
                if name in designated:
                    bars[i].set_color(dcolor)
            # end of for

            # Change the text colors of the designated names.
            for obj in ax.get_yticklabels():
                name = obj.get_text()
                if name in designated:
                    obj.set_color(dcolor)
            # end of for
    # end of for

    # Make zero notation more simple.
    fig.canvas.draw()
    for ax in fig.axes:
        labels = []
        for obj in ax.get_xticklabels():
            try:
                text = obj.get_text()
                num = float(text)
            except ValueError:
                labels.append(text)
                continue

            if num == 0:
                labels.append('0')
            else:
                labels.append(text)
        # end of for
        
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(labels)
        # end of for
    # end of for

    return fig
