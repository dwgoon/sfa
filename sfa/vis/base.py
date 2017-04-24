# -*- coding: utf-8 -*-


class BaseTable(object):

    def __init__(self, colors=None):
        if not colors:
            self._colors = dict()
        else:
            self._colors = dict(colors)

        self._set_colors(colors)

        # We should set default values using properties
    # end of def __init__

    def _set_default_color(self, prop, defval):
        if prop not in self._colors:
            self._colors[prop] = defval
    # end of def

    def _set_colors(self):
        raise NotImplementedError()

    def _create_figure(self):
        raise NotImplementedError()

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