# -*- coding: utf-8 -*-

import matplotlib


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.table import Table

import numpy as np
import pandas as pd

mpl.rc('font', family='Arial')

from .base import BaseTable


class ConditionTable(BaseTable):

    def __init__(self, conds, colors={}):

        fig, ax = plt.subplots()
        super().__init__(fig, ax, colors)

        self._set_colors(colors)

        # Set references for data objects
        self._dfc = conds  # DataFrame of condition cases

        # Set the options of figure and axis
        self._fig.set_facecolor('white')
        self._ax.grid(b=False)
        self._ax.set_frame_on(False)
        self._ax.invert_yaxis()
        self._ax.xaxis.tick_top()

        # Create matplotlib's Table object
        self._tb = Table(self._ax, bbox=[0, 0, 1, 1])
        self._tb.auto_set_font_size(False)

        # Calculate size of cell (element of Table object)
        self._calculate_cell_size()

        # Create cells for table graphics
        self._add_subtables()

        # Set default values using properties
        self.table_fontsize = 4
        self.row_label_fontsize = 5
        self.column_label_fontsize = 5
        self.line_width = 0.5

        """
        Add labels using x and y axes. The above default values should be
        assigned before adding labels
        """
        self._add_row_labels()
        self._add_column_labels()

        # Add the table graphic object
        self._ax.add_table(self._tb)

    # end of def __init__

    def _set_colors(self, colors):
        """Assign default color values, if it is not defined.
        
        """
        self._set_default_color('cond_up_cell', 'blue')
        self._set_default_color('cond_dn_cell', 'white')

    def _calculate_cell_size(self):
        # Get the size of the DataFrame
        self._n_conds = self._dfc.shape[0]  # the number of condition cases
        self._n_cond_cols = self._dfc.shape[1]  # the number of condition labels

        # The number of total rows and columns of Table
        self._nrows = self._dfc.shape[0]
        self._ncols = self._dfc.shape[1]

        # Calculate the width and height of a single cell
        L = 1.0
        H = 1.0

        self._w_cell = L / self._ncols
        self._h_cell = H / self._nrows

    def _add_subtables(self):
        self._add_condition_subtable()

    def _add_condition_subtable(self):
        for (i, j), val in np.ndenumerate(self._dfc):
            if val != 0:
                fcolor = self._colors['cond_up_cell']
            else:
                fcolor = self._colors['cond_dn_cell']

            self._tb.add_cell(i, j,
                             self._w_cell, self._h_cell,
                             loc='center', facecolor=fcolor)
    # end of def

    def _add_row_labels(self):
        """
        Add column labels using y-axis
        """
        ylabels = list(self._dfc.index)
        self._ax.yaxis.tick_left()

        yticks = [self._h_cell / 2.0,]  # The position of the first label

        # The row labels of condition subtable
        for j in range(1, self._n_conds):
            yticks.append(yticks[j-1] + self._h_cell)

        self._ax.set_yticks(yticks)
        self._ax.set_yticklabels(ylabels, minor=False)
        self._ax.tick_params(axis='y', which='major', pad=3)

        # Hide the small bars of ticks
        for tick in self._ax.yaxis.get_major_ticks():
            tick.tick1On = False
            tick.tick2On = False
    # end of def

    def _add_column_labels(self):
        """
        Add column labels using x-axis
        """
        xlabels = list(self._dfc.columns)
        xticks = [self._w_cell / 2.0,]  # The position of the first label
        # The column labels of condition subtable
        for j in range(1, self._n_cond_cols):
            xticks.append(xticks[j - 1] + self._w_cell)

        self._ax.set_xticks(xticks)
        self._ax.set_xticklabels(xlabels, rotation=90, minor=False)
        self._ax.tick_params(axis='x', which='major', pad=3)

        # Hide the small bars of ticks
        for tick in self._ax.xaxis.get_major_ticks():
            tick.tick1On = False
            tick.tick2On = False
    # end of def

    # Properties
    @property
    def table(self):
        return self._tb

    @property
    def table_fontsize(self):
        return self._table_fontsize

    @table_fontsize.setter
    def table_fontsize(self, val):
        """
        Resize text fonts
        """
        self._table_fontsize = val
        self._tb.set_fontsize(val)

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
    def line_width(self):
        return self._line_width

    @line_width.setter
    def line_width(self, val):
        """
        Adjust the width of table lines
        """
        self._line_width = val
        celld = self._tb.get_celld()
        for (i, j), cell in celld.items():
            cell.set_linewidth(self._line_width)
