# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from .base import BaseTable


class HeatmapTable(BaseTable):

    def __init__(self, df, args, fmt='.3f',
                 cmap=None, vmin=0.0, vmax=1.0,
                 annot=True, **kwargs):

        #fig, ax = plt.subplots()
        #super().__init__(fig, ax, colors)

        super().__init__(*args, **kwargs)

        # # Assign default color values, if it is not defined.
        self._set_default_color('table_edge_color', 'black')
        self._set_default_color('colorbar_edge_color', 'black')

        # Set references for data objects
        self._df = df  # A dataFrame

        self._create_figure_and_axes()
        sns.heatmap(self._df,
                    ax=self._ax,
                    annot=annot,
                    fmt=fmt,
                    annot_kws={"size": 10},
                    linecolor=self._colors['table_edge_color'],
                    vmin=vmin, vmax=vmax,
                    cmap=cmap,
                    cbar_kws={"orientation": "vertical",
                              "pad": 0.02, })

        self._qm = self._ax.collections[0]
        self._qm.set_edgecolor('face')

        # Change the labelsize
        cb = self._qm.colorbar
        cb.outline.set_linewidth(0.5)
        cb.outline.set_edgecolor(self._colors['table_edge_color'])
        cb_ax_asp = cb.ax.get_aspect()
        cb.ax.set_aspect(cb_ax_asp * 2.0)

        self._ax.xaxis.tick_top()
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        self._ax.tick_params(axis='x', which='major', pad=3)
        self._ax.tick_params(axis='y', which='major', pad=3)


        # Hide axis labels
        self._ax.set_xlabel('')
        self._ax.set_ylabel('')

        # Text element of the heatmap object
        self._texts = []
        ch = self._ax.get_children()
        for child in ch:
            if isinstance(child, matplotlib.text.Text):
                if child.get_text() != '':
                    self._texts.append(child)

        # Set default values using properties
        self.row_label_fontsize = 10
        self.column_label_fontsize = 10
        self.colorbar_label_fontsize = 10
        self.line_width = 0.5

    # end of __init__

    def _create_figure_and_axes(self):
        fig, ax = plt.subplots()
        self._fig = fig
        self._ax = ax

    # Properties
    @property
    def table_fontsize(self):
        return self._table_fontsize


    @table_fontsize.setter
    def table_fontsize(self, val):
        """
        Resize text fonts
        """
        self._table_fontsize = val
        for t in self._texts:
            t.set_fontsize(val)


    @property
    def column_label_fontsize(self):
        return self._column_label_fontsize


    @column_label_fontsize.setter
    def column_label_fontsize(self, val):
        self._column_label_fontsize = val
        self._ax.tick_params(axis='x', which='major',
                             labelsize=self._column_label_fontsize)


    @property
    def row_label_fontsize(self):
        return self._row_label_fontsize


    @row_label_fontsize.setter
    def row_label_fontsize(self, val):
        self._row_label_fontsize = val
        self._ax.tick_params(axis='y', which='major',
                             labelsize=self._row_label_fontsize)

    @property
    def colorbar_label_fontsize(self):
        return self._colorbar_label_fontsize


    @row_label_fontsize.setter
    def colorbar_label_fontsize(self, val):
        self._colorbar_label_fontsize = val
        self._qm.colorbar.ax.tick_params(axis='y', labelsize=val)


    @property
    def line_width(self):
        return self._line_width


    @line_width.setter
    def line_width(self, val):
        """
        Adjust the width of table lines
        """
        self._line_width = val
        self._qm.set_linewidth(self._line_width)


    # Read-only properties
    @property
    def colorbar(self):
        return self._qm.colorbar