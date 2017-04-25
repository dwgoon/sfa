# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import pandas as pd

mpl.rc('font', family='Arial')

from .tableaxis import ResultTableAxis
from .table_condition import ConditionTable


class BatchResultTable(ConditionTable):

    # def __init__(self, data, cons, wgap=0.01, colors={}):
    def __init__(self, data, cons, *args, **kwargs):
        # Set references for data objects
        self._dfe = data.df_exp  # DataFrame of experiment results
        self._dfr = cons  # DataFrame of consensus between exp. and sim.
        super().__init__(data.df_conds, *args, **kwargs)
    # end of def __init__

    def _create_figure_and_axes(self):
        self._fig = plt.figure()
        self._gridspec = gridspec.GridSpec(1, 2,
                                           wspace=0.01, hspace=0.0,
                                           width_ratios=[1, 1],
                                           height_ratios=[1, 1])

    def _parse_kwargs(self, **kwargs):
        """Parse the parameters for GridSpec
        """
        self._dim = kwargs.get('dim', (1, 2))
        self._wspace = kwargs.get('wspace', 0.05)
        self._hspace = kwargs.get('hspace', 0)

        ncols_conds = self._dfc.shape[1]
        ncols_res = self._dfr.shape[1]
        self._width_ratios = kwargs.get('width_ratios', [ncols_conds,
                                                         ncols_res])
        self._height_ratios = kwargs.get('height_ratios', [1])

    def _create_axes(self):
        self._axes = []
        ax_conds = self._fig.add_subplot(self._gridspec[0, 0])
        self._axes.append(ax_conds)

        ax_res = self._fig.add_subplot(self._gridspec[0, 1])
        self._axes.append(ax_res)

        for ax in self._axes:
            ax.grid(b=False)
            ax.set_frame_on(False)
            ax.invert_yaxis()
            ax.xaxis.tick_top()

    def _create_tables(self):
        super()._create_tables()
        tb = ResultTableAxis(self._axes[1],
                             self._dfr,
                             self._dfe,
                             self._colors)
        tb.fontsize = 4
        tb.linewidth = 0.5
        self._tables.append(tb)

    def _set_colors(self, colors):
        super()._set_colors(colors)
        self._set_default_color('result_up_cell', '#35FF50')  # Green
        self._set_default_color('result_dn_cell', '#FF0000')  # Red
        self._set_default_color('result_up_text', 'black')
        self._set_default_color('result_dn_text', 'white')

    def _add_labels(self):
        super()._add_labels()

        # The second table, which is result DataFrame, adds
        # column labels only.
        tb = self._tables[1]
        tb.add_column_labels()


    #def _create_figure(self):


    # def _create_axes(self):
    #     self._axes = []
    #
    #     ax_conds = self._fig.add_subplot(self._gridspec[0, 0])
    #     self._axes.append(ax_conds)
    #
    #     ax_res = self._fig.add_subplot(self._gridspec[0, 1])
    #
    #
    #     ax.grid(b=False)
    #     ax.set_frame_on(False)
    #     ax.invert_yaxis()
    #     ax.xaxis.tick_top()


    # def _calculate_cell_size(self):
    #     # Get the sizes of the DataFrames
    #     self._n_conds = self._dfc.shape[0]  # the number of condition cases
    #     self._n_cond_cols = self._dfc.shape[1]  # the number of condition labels
    #     self._n_res_cols = self._dfr.shape[1]  # the number of result labels
    #
    #     # The number of total rows and columns of Table
    #     self._nrows = self._dfc.shape[0]
    #     self._ncols = self._dfc.shape[1] + self._dfr.shape[1] + 1
    #
    #     # Calculate the width and height of a single cell
    #     L = 1.0
    #     H = 1.0
    #
    #     self._w_cell = (L - self._wgap) / (self._ncols - 1)
    #     self._h_cell = H / self._nrows
    #
    # def _add_subtables(self):
    #     """Create cells for table graphics
    #     """
    #     self._add_vertical_gap()
    #     self._add_condition_subtable()
    #     self._add_result_subtable()
    #
    # def _add_condition_subtable(self):
    #     for (i, j), val in np.ndenumerate(self._dfc):
    #         if val != 0:
    #             fcolor = self._colors['cond_up_cell']
    #         else:
    #             fcolor = self._colors['cond_dn_cell']
    #
    #         self._tb_conds.add_cell(i, j,
    #                                 self._w_cell, self._h_cell,
    #                                 loc='center', facecolor=fcolor)
    # # end of def
    #
    # def _add_result_subtable(self):
    #     """
    #     Agreement between the results of simulation and experiment
    #     """
    #     for (i, j), val_cons in np.ndenumerate(self._dfr):
    #         if val_cons == True:
    #             fcolor =  self._colors['result_up_cell']
    #         elif val_cons == False:
    #             fcolor = self._colors['result_dn_cell']
    #         else:
    #             raise ValueError("The value of result table element "
    #                              "should be bool.")
    #
    #         val_exp = self._dfe.iloc[i, j]
    #         if val_exp > 0:
    #             text_arrow = 'UP'
    #         elif val_exp < 0:
    #             text_arrow = 'DN'
    #         else:  # val_Exp == 0
    #             text_arrow = '─'
    #
    #         self._tb_conds.add_cell(i, self._n_cond_cols + 1 + j,
    #                                 self._w_cell, self._h_cell,
    #                                 text=text_arrow,
    #                                 loc='center',
    #                                 facecolor=fcolor)
    #     # end of for
    #
    #     # Set colors of the result
    #     celld = self._tb_conds.get_celld()
    #     for (i, j), val in np.ndenumerate(self._dfr):
    #         cell = celld[(i, self._n_cond_cols + 1 + j)]
    #
    #         # Set colors
    #         if val > 0:
    #             tcolor = self._colors['result_up_text']
    #         else:
    #             tcolor = self._colors['result_dn_text']
    #         # end of if-else
    #
    #         # Adjust text
    #         cell.set_text_props(color=tcolor,
    #                             weight='bold')
    #     # end of for
    #
    # # end of def
    #
    # def _add_vertical_gap(self):
    #     """
    #     Add a vertical gap between condition and result
    #     """
    #
    #     for i in range(-1, self._n_conds):
    #         self._tb_conds.add_cell(i, self._n_cond_cols, self._wgap, self._h_cell,
    #                                 text='',
    #                                 loc='center',
    #                                 edgecolor='none',
    #                                 facecolor='none')
    #
    #     # end of for
    # # end of def
    #
    # def _add_column_labels(self):
    #     """
    #     Add column labels using x-axis
    #     """
    #     xlabels = list(self._dfc.columns) + [''] + list(self._dfr.columns)
    #     xticks = [self._w_cell / 2.0,]  # The position of the first label
    #     # The column labels of condition subtable
    #     for j in range(1, self._n_cond_cols):
    #         xticks.append(xticks[j - 1] + self._w_cell)
    #
    #     # Position of the gap
    #     xticks.append(xticks[self._n_cond_cols-1] \
    #                   + self._w_cell/2.0 + self._wgap/2.0)
    #
    #     # Position of the first label in the result table
    #     xticks.append(xticks[self._n_cond_cols] \
    #                   + self._w_cell/2.0 + self._wgap/2.0)
    #
    #     # The columns of result subtable
    #     for j in range(1, self._n_res_cols):
    #         xticks.append(xticks[self._n_cond_cols + j] + self._w_cell)
    #
    #     self._ax.set_xticks(xticks)
    #     self._ax.set_xticklabels(xlabels, rotation=90, minor=False)
    #     self._ax.tick_params(axis='x', which='major', pad=3)
    #
    #     # Hide the small bars of ticks
    #     for tick in self._ax.xaxis.get_major_ticks():
    #         tick.tick1On = False

