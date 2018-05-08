# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from .base import BaseGridPlot


class Heatmap(BaseGridPlot):

    def __init__(self, df, *args, fmt='.3f',
                 cmap=None, vmin=0.0, vmax=1.0,
                 annot=True, **kwargs):

        super().__init__(*args, **kwargs)

        # Set references for data objects
        self._df = df  # A dataFrame
        sns.heatmap(self._df,
                    ax=self._axes['base'],
                    annot=annot,
                    fmt=fmt,
                    annot_kws={"size": 10},
                    linecolor=self._colors['table_edge_color'],
                    vmin=vmin, vmax=vmax,
                    cmap=cmap,
                    cbar_kws={"orientation": "vertical",
                              "pad": 0.02, })


        self._qm = self._axes['heatmap'].collections[0]
        #self._qm.set_edgecolor(self._colors['table_edge_color'])

        # Change the labelsize
        cb = self._qm.colorbar
        cb.outline.set_linewidth(0.5)
        cb.outline.set_edgecolor(self._colors['table_edge_color'])
        cb_ax_asp = cb.ax.get_aspect()
        cb.ax.set_aspect(cb_ax_asp * 2.0)

        # Remove inner lines
        children = cb.ax.get_children()
        for child in children:
            if isinstance(child, matplotlib.collections.LineCollection):
                child.set_linewidth(0)

        self._axes['heatmap'].xaxis.tick_top()
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        self._axes['heatmap'].tick_params(axis='x', which='major', pad=-2)
        self._axes['heatmap'].tick_params(axis='y', which='major', pad=3)

        # Hide axis labels
        self._axes['heatmap'].set_xlabel('')
        self._axes['heatmap'].set_ylabel('')

        # Text element of the heatmap object
        self._texts = []
        ch = self._axes['heatmap'].get_children()
        for child in ch:
            if isinstance(child, matplotlib.text.Text):
                if child.get_text() != '':
                    self._texts.append(child)

        # Set default values using properties
        self.row_tick_fontsize = 10
        self.column_tick_fontsize = 10
        self.colorbar_label_fontsize = 10
        self.linewidth = 0.5

    # end of __init__

    def _set_colors(self, colors):
        """Assign default color values for heatmap and colorbar
        """
        self._set_default_color('table_edge_color', 'black')
        self._set_default_color('colorbar_edge_color', 'black')

    def _create_axes(self):
        super()._create_axes()
        ax = self._axes['base']
        self._axes['heatmap'] = ax
        #del self._axes['base']

    # Properties
    @property
    def text_fontsize(self):
        return self._text_fontsize

    @text_fontsize.setter
    def text_fontsize(self, val):
        """Resize text fonts
        """
        self._text_fontsize = val
        for t in self._texts:
            t.set_fontsize(val)

    @property
    def column_tick_fontsize(self):
        return self._column_tick_fontsize

    @column_tick_fontsize.setter
    def column_tick_fontsize(self, val):
        self._column_tick_fontsize = val
        self._axes['heatmap'].tick_params(
                                axis='x',
                                which='major',
                                labelsize=self._column_tick_fontsize)

    @property
    def row_tick_fontsize(self):
        return self._row_tick_fontsize

    @row_tick_fontsize.setter
    def row_tick_fontsize(self, val):
        self._row_tick_fontsize = val
        self._axes['heatmap'].tick_params(
                                 axis='y',
                                 which='major',
                                 labelsize=self._row_tick_fontsize)

    @property
    def colorbar_tick_fontsize(self):
        return self._colorbar_tick_fontsize

    @colorbar_tick_fontsize.setter
    def colorbar_tick_fontsize(self, val):
        self._colorbar_tick_fontsize = val
        self._qm.colorbar.ax.tick_params(axis='y', labelsize=val)

    @property
    def linewidth(self):
        return self._linewidth

    @linewidth.setter
    def linewidth(self, val):
        """Adjust the width of table lines
        """
        self._linewidth = val
        self._qm.set_linewidth(self._linewidth)

    # Read-only properties
    @property
    def colorbar(self):
        return self._qm.colorbar