
import numpy as np
import scipy.spatial.distance as distance
import scipy.cluster.hierarchy as sch

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns

from .table_condition import ConditionTable


class HierarchicalClusteringTable(ConditionTable):

    def __init__(self, conds,  samples, *args, **kwargs):
        # Set references for data objects
        self._dfs = samples  # DataFrame of samples to be clustered.
        super().__init__(conds, *args, **kwargs)
        self._create_colorbar()
        self.column_tick_fontsize = self._table_tick_fontsize
    # end of def __init__

    def _parse_kwargs(self, **kwargs):
        """Parse the keyword arguments.
        """

        self._vmin = kwargs.get('vmin', None)
        self._vmax = kwargs.get('vmax', None)

        colors_blend = ['red', 'white', np.array([0, 1, 0, 1])]
        default_cmap = sns.blend_palette(colors_blend,
                                         n_colors=100,
                                         as_cmap=True)
        self._cmap = kwargs.get('cmap', default_cmap)

        self._dim = kwargs.get('dim', (2, 5))
        self._wspace = kwargs.get('wspace', 0.005)
        self._hspace = kwargs.get('hspace', 0.005)

        default_width_ratios = [self._dfc.shape[1],
                                0.25,
                                self._dfs.shape[1],
                                0.5*self._dfs.shape[1],
                                0.05*self._dfs.shape[1]]

        default_height_ratios = [self._dfs.shape[1],
                                 self._dfc.shape[0]]

        self._width_ratios = kwargs.get('width_ratios',
                                        default_width_ratios)
        self._height_ratios = kwargs.get('height_ratios',
                                         default_height_ratios)

        default_position = {'condition': np.array([1, 0]),
                            'heatmap': np.array([1, 2]),
                            'row_dendrogram': np.array([1, 3]),
                            'col_dendrogram': np.array([0, 2]),
                            'colorbar': np.array([1, 4])}

        self._axes_position = kwargs.get('axes_position',
                                         default_position)

        self._row_cluster = kwargs.get('row_cluster', True)
        self._col_cluster = kwargs.get('col_cluster', True)

        if self._row_cluster:
            self._row_method = kwargs.get('row_method', 'single')
            self._row_metric = kwargs.get('row_metric', 'cityblock')
            self._row_dend_linewidth = kwargs.get('row_dend_linewidth', 0.5)

        if self._col_cluster:
            self._col_method = kwargs.get('col_method', 'single')
            self._col_metric = kwargs.get('col_metric', 'cityblock')
            self._col_dend_linewidth = kwargs.get('col_dend_linewidth', 0.5)

        self._table_linewidth = kwargs.get('table_linewidth', 0.5)
        self._table_tick_fontsize = kwargs.get('table_tick_fontsize', 5)
        self._colorbar_tick_fontsize = kwargs.get('colorbar_tick_fontsize', 5)

    def _create_axes(self):
        self._axes = {}
        # pos = self._axes_position['heatmap']
        # ax_heatmap = self._fig.add_subplot(self._gridspec[pos[0], pos[1]])
        # self._axes.append(ax_heatmap)

        pos = self._axes_position['condition']
        ax_conds = self._fig.add_subplot(self._gridspec[pos[0], pos[1]])
        self._axes['condition'] = ax_conds
        ax_conds.grid(b=False)
        ax_conds.set_frame_on(False)
        ax_conds.invert_yaxis()
        ax_conds.xaxis.tick_bottom()

        self._perform_clustering()

    def _create_tables(self):
        super()._create_tables()
        ax_heatmap = self._axes['heatmap']

        # Draw lines on table and heatmap
        self.tables[0].linewidth = self._table_linewidth
        for x in range(self._dfs.shape[1]+1):
            ax_heatmap.axvline(x-0.5,
                               linewidth=self._table_linewidth,
                               color='k', zorder=10)

        for y in range(self._dfs.shape[0]+1):
            ax_heatmap.axhline(y-0.5,
                               linewidth=self._table_linewidth,
                               color='k', zorder=10)

    def _perform_clustering(self):
        sch.set_link_color_palette(['black'])
        if self._row_cluster:

            row_pairwise_dists = distance.pdist(self._dfs,
                                                metric=self._row_metric)
            row_clusters = sch.linkage(row_pairwise_dists,
                                       metric=self._row_metric,
                                       method=self._row_method)

            with plt.rc_context({'lines.linewidth': self._row_dend_linewidth}):
                # Dendrogram for row clustering
                pos = self._axes_position['row_dendrogram']
                subgs = self._gridspec[pos[0], pos[1]]
                ax_row_den = self._fig.add_subplot(subgs)
                row_den = sch.dendrogram(row_clusters,
                                         color_threshold=np.inf,
                                         orientation='right')

                ax_row_den.set_facecolor("white")
                self._clean_axis(ax_row_den)
                self._axes['row_dendrogram'] = ax_row_den

                ind_row = row_den['leaves']
                # Rearrange the DataFrame for condition according to
                # the clustering result.
                self._dfc = self._dfc.iloc[ind_row, :]
        else:
            ind_row = range(self._dfs.index.size)  #self._dfs.index.ravel()

        if self._col_cluster:
            col_pairwise_dists = distance.pdist(self._dfs.T,
                                                metric=self._col_metric)
            col_clusters = sch.linkage(col_pairwise_dists,
                                       metric=self._col_metric,
                                       method=self._col_method)

            with plt.rc_context({'lines.linewidth': self._col_dend_linewidth}):
                # Dendrogram for column clustering
                pos = self._axes_position['col_dendrogram']
                ax_col_den = self._fig.add_subplot(self._gridspec[pos[0], pos[1]])
                col_den = sch.dendrogram(col_clusters,
                                         color_threshold=np.inf,
                                         orientation='top')
                ax_col_den.set_facecolor("white")
                self._clean_axis(ax_col_den)
                self._axes['col_dendrogram'] = ax_col_den
                ind_col = col_den['leaves']
        else:
            # ind_col = self._dfs.columns.ravel()
            ind_col = range(self._dfs.columns.size)

        # Heatmap
        pos = self._axes_position['heatmap']
        subgs = self._gridspec[pos[0], pos[1]]
        ax_heatmap = self._fig.add_subplot(subgs)
        self._heatmap = ax_heatmap.matshow(self._dfs.iloc[ind_row, ind_col],
                                           vmin=self._vmin,
                                           vmax=self._vmax,
                                           interpolation='nearest',
                                           aspect='auto',
                                           #origin='lower',
                                           cmap=self._cmap)

        ax_heatmap.grid(b=False)
        ax_heatmap.set_frame_on(True)
        ax_heatmap.xaxis.tick_bottom()
        self._axes['heatmap'] = ax_heatmap
        self._clean_axis(ax_heatmap)

        # Remove the y-labels of condition table
        #ax_conds = self._axes['condition']

        # Add column labels
        ax_heatmap.set_xticks(np.arange(0, self._dfs.shape[1], 1))
        ax_heatmap.set_xticklabels(np.array(self._dfs.columns[ind_col]),
                                   rotation=90, minor=False)

        ax_heatmap.tick_params(axis='x', which='major', pad=-2)

        # Remove the tick lines
        for line in ax_heatmap.get_xticklines():
            line.set_markersize(0)

        for line in ax_heatmap.get_yticklines():
            line.set_markersize(0)

    def _create_colorbar(self):
        pos = self._axes_position['colorbar']
        subgs = self._gridspec[pos[0], pos[1]]
        ax_colorbar = self._fig.add_subplot(subgs)

        cb = self._fig.colorbar(self._heatmap, ax_colorbar,
                                drawedges=False)  #True)
        self._colorbar = cb
        #cb.ax.yaxis.set_ticks_position('right')
        #self._clean_axis(cb.ax)
        # for sp in cb.ax.spines.values():
        #     sp.set_visible(False)
        cb.ax.yaxis.set_ticks_position('none')
        cb.ax.yaxis.set_tick_params(pad=-2)
        cb.ax.yaxis.set_label_position('right')
        cb.outline.set_edgecolor('black')
        cb.outline.set_linewidth(self._table_linewidth)
        self.colorbar_fontsize = self._colorbar_tick_fontsize

    def _clean_axis(self, ax):
        """Remove ticks, tick labels, and frame from axis
        """
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    def _add_labels(self):
        """Add only column labels for condition table. 
        """
        tb = self._tables[0]
        tb.add_column_labels()

    @property
    def table_linewidth(self):
        return self._table_linewidth

    @table_linewidth.setter
    def table_linewidth(self, val):
        self._table_linewidth = val

    @property
    def colorbar(self):
        return self._colorbar

    @property
    def colorbar_fontsize(self):
        return self._colorbar_tick_fontsize

    @colorbar_fontsize.setter
    def colorbar_fontsize(self, val):
        self._colorbar_tick_fontsize = val
        ticks = self._colorbar.ax.yaxis.get_ticklabels()
        for t in ticks:
            t.set_fontsize(self._colorbar_tick_fontsize)

    # @column_tick_fontsize.setter
    # def column_tick_fontsize(self, val):
    #     self._column_tick_fontsize = val
    #     for ax in self._axes:
    #         ax.tick_params(axis='x', which='major',
    #                        labelsize=self._column_tick_fontsize)

    # def _set_colors(self, colors):
    #     super()._set_colors(colors)
    #
    # def _add_labels(self):
    #     super()._add_labels()  # Add labels for condition table


    # def _add_column_labels(self):
    #     """Add column labels using x-axis
    #     """
    #     xlabels = list(self._dfs.columns)
    #     ax_heatmap.set_xticks(np.arange(self._dfs.shape[0]))
    #     ax_heatmap.set_xticklabels(xlabels,
    #                                rotation=90, minor=False)
    #     ax_heatmap.tick_params(axis='x', which='major', pad=3)
    #
    #     # Hide the small bars of ticks
    #     for tick in self._ax.xaxis.get_major_ticks():
    #         tick.tick1On = False
    #         tick.tick2On = False
    # # # end of def