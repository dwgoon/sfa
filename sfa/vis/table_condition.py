# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.table import Table
from matplotlib.gridspec import GridSpec


import numpy as np
import pandas as pd

mpl.rc('font', family='Arial')

from .base import BaseTable
from .tableaxis import ConditionTableAxis


class ConditionTable(BaseTable):

    def __init__(self, conds, *args, **kwargs):
        self._dfc = conds  # DataFrame of condition cases
        super().__init__(*args, **kwargs)
        """
        Add labels using x and y axes.
        The default values should be assigned before adding labels.
        """
        self.row_label_fontsize = 5
        self.column_label_fontsize = 5
        self._add_labels()
    # end of def __init__

    def _set_colors(self, colors):
        """Assign default color values, if it is not defined.
        """
        self._set_default_color('cond_up_cell', 'blue')
        self._set_default_color('cond_dn_cell', 'white')

    def _create_tables(self):
        self._tables = []
        tb = ConditionTableAxis(self._axes[0], self._dfc, self._colors)
        tb.fontsize = 4
        tb.linewidth = 0.5
        self._tables.append(tb)

    def _add_labels(self):
        tb = self._tables[0]
        tb.add_row_labels()
        tb.add_column_labels()


