
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table


class TableAxis(object):
    def __init__(self, ax, df, colors):
        self._ax = ax
        self._df = df
        self._colors = colors

        # Create matplotlib's Table object
        self._tb = Table(self._ax, bbox=[0, 0, 1, 1])
        self._tb.auto_set_font_size(False)
        self._ax.add_table(self._tb)

        self._calculate_cell_size()
        self._add_cells()
        self._remove_ticks()

    def _calculate_cell_size(self):
        # The number of total rows and columns of the table
        self._nrows = self._df.shape[0]
        self._ncols = self._df.shape[1]

        # Calculate the width and height of a single cell in the table
        L = 1.0
        H = 1.0
        self._w_cell = L / self._ncols
        self._h_cell = H / self._nrows

    def _add_cells(self):
        raise NotImplementedError()

    def _remove_ticks(self):
        self._ax.get_xaxis().set_ticks([])
        self._ax.get_yaxis().set_ticks([])

    def add_row_labels(self):
        """Add row labels using y-axis
        """
        ylabels = list(self._df.index)
        self._ax.yaxis.tick_left()

        yticks = [self._h_cell/2.0]  # The position of the first label

        # The row labels of condition subtable
        for j in range(1, self._nrows):
            yticks.append(yticks[j-1] + self._h_cell)

        self._ax.set_yticks(yticks)
        self._ax.set_yticklabels(ylabels, minor=False)
        self._ax.tick_params(axis='y', which='major', pad=3)

        # Hide the small bars of ticks
        for tick in self._ax.yaxis.get_major_ticks():
            tick.tick1On = False
            tick.tick2On = False
    # end of def

    def add_column_labels(self):
        """
        Add column labels using x-axis
        """
        xlabels = list(self._df.columns)
        xticks = [self._w_cell/2.0]  # The position of the first label
        # The column labels of condition subtable
        for j in range(1, self._ncols):
            xticks.append(xticks[j - 1] + self._w_cell)

        self._ax.xaxis.set_ticks_position('none')

        # # Hide the small bars of ticks
        # for tick in self._ax.xaxis.get_major_ticks():
        #     tick.tick1On = False
        #     tick.tick2On = False

        self._ax.set_xticks(xticks)
        self._ax.set_xticklabels(xlabels, rotation=90, minor=False)
        self._ax.tick_params(axis='x', which='major', pad=-2)

    # end of def

    @property
    def fontsize(self):
        return self._table_fontsize

    @fontsize.setter
    def fontsize(self, val):
        """
        Resize text fonts
        """
        self._table_fontsize = val
        self._tb.set_fontsize(val)

    @property
    def linewidth(self):
        return self._linewidth

    @linewidth.setter
    def linewidth(self, val):
        """Adjust the width of table lines
        """
        self._linewidth = val
        celld = self._tb.get_celld()
        for (i, j), cell in celld.items():
            cell.set_linewidth(self._linewidth)


class ConditionTableAxis(TableAxis):

    def _add_cells(self):
        for (i, j), val in np.ndenumerate(self._df):
            if val != 0:
                fcolor = self._colors['cond_up_cell']
            else:
                fcolor = self._colors['cond_dn_cell']

            self._tb.add_cell(i, j,
                              self._w_cell, self._h_cell,
                              loc='center', facecolor=fcolor)


class ResultTableAxis(TableAxis):
    
    def __init__(self, ax, dfr, dfe, colors):
        self._dfr = dfr
        self._dfe = dfe
        super().__init__(ax, dfr, colors)

    def _add_cells(self):
        """Add cells for representing the agreement between
        the results of simulation and experiment.
        """
        for (i, j), val_cons in np.ndenumerate(self._dfr):
            if val_cons == True:
                fcolor =  self._colors['result_up_cell']
            elif val_cons == False:
                fcolor = self._colors['result_dn_cell']
            else:
                raise ValueError("The value of result table element "
                                 "should be bool.")

            val_exp = self._dfe.iloc[i, j]
            if val_exp > 0:
                text_arrow = 'UP'
            elif val_exp < 0:
                text_arrow = 'DN'
            else:  # val_Exp == 0
                text_arrow = '─'

            self._tb.add_cell(i, j,
                              self._w_cell, self._h_cell,
                              text=text_arrow,
                              loc='center',
                              facecolor=fcolor)
        # end of for

        # Set colors of the result
        celld = self._tb.get_celld()
        for (i, j), val in np.ndenumerate(self._dfr):
            cell = celld[(i, j)]

            # Set colors
            if val > 0:
                tcolor = self._colors['result_up_text']
            else:
                tcolor = self._colors['result_dn_text']
            # end of if-else

            # Adjust text
            cell.set_text_props(color=tcolor,
                                weight='bold')
        # end of for