# -*- coding: utf-8 -*-

class BaseTable(object):

    def __init__(self, fig, ax, colors={}):
        self._fig = fig
        self._ax = ax
        self._colors = dict(colors)

        # We should set default values using properties
    # end of def __init__

    def _set_default_color(self, prop, defval):
        if prop not in self.colors:
            self.colors[prop] = defval
    # end of def

    # Properties
    @property
    def table_fontsize(self):
        raise NotImplementedError()

    @table_fontsize.setter
    def table_fontsize(self, val):
        raise NotImplementedError()

    @property
    def column_label_fontsize(self):
        raise NotImplementedError()

    @column_label_fontsize.setter
    def column_label_fontsize(self, val):
        raise NotImplementedError()

    @property
    def row_label_fontsize(self):
        raise NotImplementedError()

    @row_label_fontsize.setter
    def row_label_fontsize(self, val):
        raise NotImplementedError()


    @property
    def line_width(self):
        raise NotImplementedError()

    @line_width.setter
    def line_width(self, val):
        raise NotImplementedError()

    # Read-only properties
    @property
    def colors(self):
        return self._colors

    @property
    def fig(self):
        return self._fig

    @property
    def ax(self):
        return self._ax