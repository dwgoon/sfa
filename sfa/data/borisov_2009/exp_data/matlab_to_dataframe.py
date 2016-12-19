# -*- coding: utf-8 -*-

import glob
import pandas as pd

pert_nodes = ['SS', 'GAB1', 'IRS',
              'RasGAP', 'RAS', 'RAF', 'MEK',
              'PI3K', 'PIP3', 'PDK1', 'mTOR']


readout_nodes = ['SS', 'GAB1', 'IRS',
                 'RasGAP', 'RAS', 'RAF', 'MEK',
                 'PI3K', 'PIP3', 'PDK1', 'mTOR', 'ERK', 'AKT']
                 

for fname in glob.glob("*.txt"):
    print (fname)
    
    
    df = pd.read_csv(fname, sep='\s+',
                     header=None, index_col=None)
                    
                    
    df.index = df.index = readout_nodes
    df.columns = df.columns = range(1, df.shape[1]+1)

    str_file_name = fname.split('.')[0]
    df.T.to_csv("exp_%s.tsv"%(str_file_name), sep='\t')
