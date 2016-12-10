# -*- coding: utf-8 -*-

import sfa

ds = sfa.DataSet()
ds.create()

for abbr, data in ds.items():
    if not isinstance(data, sfa.base.Data):
        data = sfa.get_avalue(data)
    print (abbr)
    print ("N: ", data.dg.number_of_nodes())
    print ("L: ", data.dg.number_of_edges())
    print ("I: ", len(data.inputs))
    print ("T: ", data.df_ptb.index.size)
    print ("C: ", data.df_conds.index.size)
    print ("R: ", data.df_exp.columns.size)