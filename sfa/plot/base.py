# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class BaseGridPlot(object):

    def __init__(self, colors=None, *args, **kwargs):
        if not colors:
            self._colors = dict()
        else:
            self._colors = dict(colors)

        self._set_colors(colors)
        self._parse_kwargs(**kwargs)

        # Create figure and axis
        self._create_figure()
        self._create_axes()

    # end of def __init__

    def _set_default_color(self, prop, defval):
        if prop not in self._colors:
            self._colors[prop] = defval
    # end of def

    def _set_colors(self):
        raise NotImplementedError()

    def _parse_kwargs(self, **kwargs):
        """Parse the parameters for GridSpec
        """
        self._dim = kwargs.get('dim', (1, 1))
        self._wspace = kwargs.get('wspace', 0)
        self._hspace = kwargs.get('hspace', 0)
        self._width_ratios = kwargs.get('width_ratios', [1])
        self._height_ratios = kwargs.get('height_ratios', [1])

    def _create_figure(self):
        self._fig = plt.figure()
        self._gridspec = GridSpec(*self._dim,
                                  wspace=self._wspace,
                                  hspace=self._hspace,
                                  width_ratios=self._width_ratios,
                                  height_ratios=self._height_ratios)

        self._fig.set_facecolor('white')

    def _create_axes(self):
        self._axes = {}
        ax = self._fig.add_subplot(self._gridspec[0, 0])
        self._axes['base'] = ax
        ax.grid(b=False)
        ax.set_frame_on(False)
        ax.invert_yaxis()
        ax.xaxis.tick_top()

    @property
    def column_tick_fontsize(self):
        return self._column_tick_fontsize

    @column_tick_fontsize.setter
    def column_tick_fontsize(self, val):
        self._column_tick_fontsize = val
        for ax in self._axes.values():
            ax.tick_params(axis='x', which='major',
                           labelsize=self._column_tick_fontsize)

    @property
    def row_tick_fontsize(self):
        return self._row_tick_fontsize

    @row_tick_fontsize.setter
    def row_tick_fontsize(self, val):
        self._row_tick_fontsize = val
        for ax in self._axes.values():
            ax.tick_params(axis='y', which='major',
                           labelsize=self._row_tick_fontsize)

    # # Properties
    # @property
    # def column_tick_fontsize(self):
    #     raise NotImplementedError()
    #
    # @column_tick_fontsize.setter
    # def column_tick_fontsize(self, val):
    #     raise NotImplementedError()
    #
    # @property
    # def row_tick_fontsize(self):
    #     raise NotImplementedError()
    #
    # @row_tick_fontsize.setter
    # def row_tick_fontsize(self, val):
    #     raise NotImplementedError()

    # @property
    # def text_fontsize(self):
    #     raise NotImplementedError()
    #
    # @text_fontsize.setter
    # def text_fontsize(self, val):
    #     raise NotImplementedError()
    #
    # @property
    # def linewidth(self):
    #     raise NotImplementedError()
    #
    # @linewidth.setter
    # def linewidth(self, val):
    #     raise NotImplementedError()

    # Read-only properties
    @property
    def colors(self):
        return self._colors

    @property
    def fig(self):
        return self._fig

    @property
    def axes(self):
        return self._axes

    @property
    def gridspec(self):
        return self._gridspec


class BaseTable(BaseGridPlot):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_tables()

    def _create_tables(self):
        raise NotImplementedError()

    @property
    def tables(self):
        return self._tables