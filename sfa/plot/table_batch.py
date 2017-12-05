# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .tableaxis import ResultTableAxis
from .table_condition import ConditionTable


class BatchResultTable(ConditionTable):

    def __init__(self, data, cons, *args, **kwargs):
        # Set references for data objects
        self._dfe = data.df_exp  # DataFrame of experiment results
        self._dfr = cons  # DataFrame of consensus between exp. and sim.
        super().__init__(data.df_conds, *args, **kwargs)
    # end of def __init__

    def _parse_kwargs(self, **kwargs):
        """Parse the keyword arguments.
        """
        self._dim = kwargs.get('dim', (1, 2))
        self._wspace = kwargs.get('wspace', 0.05)
        self._hspace = kwargs.get('hspace', 0)

        ncols_conds = self._dfc.shape[1]
        ncols_res = self._dfr.shape[1]
        self._width_ratios = kwargs.get('width_ratios', [ncols_conds,
                                                         ncols_res])
        self._height_ratios = kwargs.get('height_ratios', [1])

        default_position = {'condition': (0, 0), 'result': (0, 1), }
        self._axes_position = kwargs.get('axes_position',
                                         default_position)

    def _create_axes(self):
        self._axes = {}
        pos = self._axes_position['condition']
        ax_conds = self._fig.add_subplot(self._gridspec[pos[0], pos[1]])
        self._axes['condition'] = ax_conds

        pos = self._axes_position['result']
        ax_res = self._fig.add_subplot(self._gridspec[pos[0], pos[1]])
        self._axes['result'] = ax_res

        for ax in self._axes.values():
            ax.grid(b=False)
            ax.set_frame_on(False)
            ax.invert_yaxis()
            ax.xaxis.tick_top()

    def _create_tables(self):
        super()._create_tables()
        tb = ResultTableAxis(self._axes['result'],
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