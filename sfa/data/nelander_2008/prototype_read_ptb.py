# -*- coding: utf-8 -*-

import pandas as pd


#
#df = pd.DataFrame(columns=["Target", "Type", "Value"])
#
#
#with open("ptb.tsv", "r") as fin:
#    for line in fin:
#        line = line.strip('#')
#        items = line.split()
#        print (line)
#        
        


df_ptb = pd.read_table("ptb.tsv", index_col=0)

for target, row in df_ptb.iterrows():
    #print (i, row.Target, row.Type, row.Value, row.Comment)
    print (target, row.Type, row.Value, row.Comment)