# -*- coding: utf-8 -*-

import copy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import sfa
import sfa.vis
from sfa import calc_accuracy
from sfa import AlgorithmSet
from sfa import DataSet


if __name__ == "__main__":

    # Create containers for algorithm and data.
    algs = AlgorithmSet()
    ds = DataSet()

    # Load an algorithm and a data.
    alg_name = 'APS'
    algs.create(alg_name)
    alg = algs[alg_name]

    ds.create("NELANDER_2008")
    data = ds["NELANDER_2008"]

    results = {}
    data.A[data.n2i["p-ERK"], data.n2i["p-MEK"]]
    alg.data = data
    alg.params.initialize()
    alg.params.use_rel_change = True
    alg.initialize()
    alg.compute_batch()
    acc, cons = calc_accuracy(alg.result.df_sim,
                              data.df_exp,
                              get_cons=True)

    print ("Accuracy: ", acc)



    # brt = sfa.vis.BatchResultTable(data, cons)
    # brt.column_label_fontsize = 4
    # brt.ax.tick_params(axis='x', which='both', pad=0.01)
    #
    # brt.row_label_fontsize = 4
    # brt.ax.tick_params(axis='y', which='both', pad=0.01)
    #
    # plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.8,
    #                     wspace=0.1, hspace=0.1)
    #
    # brt.fig.set_size_inches(2, 3)
    # brt.fig.savefig("%s_%s.png"%(alg.abbr, data.abbr), dpi=300)


    names_ba_se = []
    vals_ba_se = []  # 'se' means a 'single experiment'
    alg.apply_inputs(names_ba_se, vals_ba_se)
    x_cnt, paths_cnt = alg.wire(data.dg, names_ba_se, vals_ba_se,
                                 get_path=True)

    targets_ptb = ['p-MEK']
    dg_ptb = data.dg.copy()
    names_ba_se = []
    vals_ba_se = []  # 'se' means a 'single experiment'
    alg.apply_inputs(names_ba_se, vals_ba_se)
    alg.apply_perturbations(targets_ptb,
                            names_ba_se, vals_ba_se,
                            dg_ptb)

    # paths_ptb = []
    # for i, src in enumerate(ptbs):
    #    x, p = alg.wire(dg_ptb, [src], [vals[i]], get_path=True)
    #    paths_ptb.extend(p)

    x_ptb, paths_ptb = alg.wire(data.dg, names_ba_se, vals_ba_se,
                                get_path=True)


    def get_link_str(x, y):
        sign = data.dg.edge[x][y]['sign']
        if sign > 0:
            return '->'
        else:
            return '-|'

    def print_paths(dg, paths, targets):
        paths = sorted(paths, key=lambda p: p[-1])

        for p in paths:
            str_path = str(p[0])
            if p[-1] not in targets:
                continue

            for i in range(len(p) - 1):
                src = p[i]
                tgt = p[i + 1]
                str_link = get_link_str(src, tgt)
                str_path += " %s %s" % (str_link, tgt)

            if p[0] == 'EGF':
                ba = +1
            else:
                ba = -1

            F = alg.wire_single_path(dg, ba, p)
            print(F, str_path)  # end of main


    # end of def

    targets = ['p-MEK']
    print ("<Control>")
    print_paths(data.dg, paths_cnt, targets)

    print("<Experiment>")
    print_paths(dg_ptb, paths_ptb, targets)
